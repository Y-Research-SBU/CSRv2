import argparse
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SparseEncoder
from datasets import load_dataset
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Sparse Encoder model on retrieval tasks")
    parser.add_argument("--model_name", type=str, required=True, help="Model name or path for SparseEncoder")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the retrieval dataset to evaluate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--save_root_path", type=str, required=True, help="Root path to save embeddings")
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use for processing")
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 5, 10, 20, 30], help="K values for NDCG calculation")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    parser.add_argument("--topk_query", type=int, default=None, help="Keep only top-k dimensions for query embeddings (None means keep all)")
    parser.add_argument("--topk_corpus", type=int, default=None, help="Keep only top-k dimensions for corpus embeddings (None means keep all)")
    return parser.parse_args()


def dcg_at_k(relevance_scores, k):
    relevance_scores = np.array(relevance_scores)[:k]
    gains = 2**relevance_scores - 1
    discounts = np.log2(np.arange(len(relevance_scores)) + 2)
    return np.sum(gains / discounts)


def ndcg_at_k(relevance_scores, k):
    dcg = dcg_at_k(relevance_scores, k)
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_relevance_scores, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def apply_topk_sparsification(embeddings, topk):
    if topk is None:
        return embeddings
    sparsified_embeddings = np.zeros_like(embeddings)
    
    for i in range(embeddings.shape[0]):
        sample = embeddings[i]
        if topk < len(sample):
            topk_indices = np.argpartition(sample, -topk)[-topk:]
            sparsified_embeddings[i, topk_indices] = sample[topk_indices]
        else:
            sparsified_embeddings[i] = sample
    
    return sparsified_embeddings


def encode_texts_in_batches(model, texts, batch_size, is_query=True, topk=None):
    embedding_buffer = []
    num_batches = (len(texts) + batch_size - 1) // batch_size
    
    desc = f"Encoding {'queries' if is_query else 'documents'}"
    if topk is not None:
        desc += f" (top-{topk})"
    
    for i in tqdm(range(num_batches), desc=desc):
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]
        
        with torch.no_grad():
            if is_query:
                batch_embeddings = model.encode_query(batch_texts)
            else:
                batch_embeddings = model.encode_document(batch_texts)
        
        if isinstance(batch_embeddings, torch.Tensor):
            if batch_embeddings.is_sparse:
                batch_embeddings = batch_embeddings.to_dense()
            batch_embeddings = batch_embeddings.cpu().numpy()
        
        if topk is not None:
            batch_embeddings = apply_topk_sparsification(batch_embeddings, topk)
        
        embedding_buffer.append(batch_embeddings)
    
    return np.vstack(embedding_buffer) if embedding_buffer else np.array([])


def get_retrieval_embeddings(model, dataset_name, batch_size, save_root_path, topk_query=None, topk_corpus=None):
    assert dataset_name in [
        "arguana",
        "ClimateFEVER_test_top_250_only_w_correct-v2",
        "cqadupstack-gaming",
        "cqadupstack-unix",
        "fiqa",  
        "nfcorpus",
        "scidocs",
        "scifact",
    ], f"Dataset {dataset_name} not supported for retrieval"
    
    dataset = load_dataset(f"mteb/{dataset_name}")
    output_dir = os.path.join(save_root_path, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(os.path.join(output_dir, "corpus.npz")):
        print(f"Corpus for dataset {dataset_name} already processed, skipping...")
    else:
        try:
            corpus_dataset = load_dataset("mteb/" + dataset_name, name="corpus")["corpus"]
        except:
            corpus_dataset = load_dataset("mteb/" + dataset_name, name="corpus")["test"]        
        print("Processing corpus...")
        corpus_texts = [line["title"] + " " + line["text"] for line in tqdm(corpus_dataset)]
        corpus_ids = [line["_id"] for line in corpus_dataset]
        
        corpus_embeddings = encode_texts_in_batches(model, corpus_texts, batch_size, is_query=False, topk=topk_corpus)
        
        np.savez(os.path.join(output_dir, "corpus.npz"), data=corpus_embeddings, id=corpus_ids)
        print(f"Completed processing corpus, saved {len(corpus_embeddings)} embeddings")
        if topk_corpus is not None:
            print(f"Applied top-{topk_corpus} sparsification to corpus embeddings")
    
    if os.path.exists(os.path.join(output_dir, "queries.npz")):
        print(f"Queries for dataset {dataset_name} already processed, skipping...")
    else:
        try:
            query_dataset = load_dataset("mteb/" + dataset_name, name="queries")["queries"]
        except:
            try:
                query_dataset = load_dataset("mteb/" + dataset_name, name="queries")["train"]
            except:
                query_dataset = load_dataset("mteb/" + dataset_name, name="queries")["test"]    
        print("Processing queries...")
        query_texts = [line["text"] for line in tqdm(query_dataset)]
        query_ids = [line["_id"] for line in query_dataset]
        
        queries_embeddings = encode_texts_in_batches(model, query_texts, batch_size, is_query=True, topk=topk_query)
        
        np.savez(os.path.join(output_dir, "queries.npz"), data=queries_embeddings, id=query_ids)
        print(f"Completed processing queries, saved {len(queries_embeddings)} embeddings")
        if topk_query is not None:
            print(f"Applied top-{topk_query} sparsification to query embeddings")

    # Process splits (relations)
    for split in dataset.keys():
        split_dataset = dataset[split]
        query_id_buffer = [line["query-id"] for line in split_dataset]
        corpus_id_buffer = [line["corpus-id"] for line in split_dataset]
        
        np.savez(os.path.join(output_dir, f"{split}.npz"),
                 query_id=query_id_buffer, corpus_id=corpus_id_buffer)
        print(f"Completed processing {split} relations, saved {len(query_id_buffer)} pairs")


def calculate_ndcg_from_embeddings(embeddings_path, dataset_name, split="test", k_values=[1, 5, 10, 20, 30], use_gpu=True, batch_size=64):
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    
    if isinstance(k_values, int):
        k_values = [k_values]
    
    dataset_path = os.path.join(embeddings_path, dataset_name)
    
    corpus_data = np.load(os.path.join(dataset_path, "corpus.npz"))
    queries_data = np.load(os.path.join(dataset_path, "queries.npz"))
    relation_data = np.load(os.path.join(dataset_path, f"{split}.npz"))
    
    corpus_embeddings = corpus_data["data"]  # [num_corpus, vocab_size]
    corpus_ids = corpus_data["id"]
    queries_embeddings = queries_data["data"]  # [num_queries, vocab_size]
    queries_ids = queries_data["id"]
    query_ids_rel = relation_data["query_id"]
    corpus_ids_rel = relation_data["corpus_id"]
    
    corpus_embeddings_tensor = torch.from_numpy(corpus_embeddings).float().to(device)
    queries_embeddings_tensor = torch.from_numpy(queries_embeddings).float().to(device)
    
    corpus_id_to_idx = {id_: idx for idx, id_ in enumerate(corpus_ids)}
    query_id_to_idx = {id_: idx for idx, id_ in enumerate(queries_ids)}
    
    relevance_dict = {}
    for query_id, corpus_id in zip(query_ids_rel, corpus_ids_rel):
        if query_id not in relevance_dict:
            relevance_dict[query_id] = set()
        relevance_dict[query_id].add(corpus_id)
    
    valid_query_ids = [qid for qid in relevance_dict.keys() if qid in query_id_to_idx]
    query_indices = [query_id_to_idx[qid] for qid in valid_query_ids]
    
    ndcg_scores_dict = {k: [] for k in k_values}
    
    num_queries = len(query_indices)
    num_batches = (num_queries + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Computing NDCG batches"):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_queries)
        batch_query_indices = query_indices[batch_start:batch_end]
        batch_query_ids = valid_query_ids[batch_start:batch_end]
        
        batch_queries_tensor = queries_embeddings_tensor[batch_query_indices]
        
        similarities_batch = torch.mm(batch_queries_tensor, corpus_embeddings_tensor.t())
        sorted_indices_batch = torch.argsort(similarities_batch, dim=1, descending=True)
        sorted_indices_batch_cpu = sorted_indices_batch.cpu().numpy()
        
        for i, query_id in enumerate(batch_query_ids):
            sorted_corpus_indices = sorted_indices_batch_cpu[i]
            sorted_corpus_ids = [corpus_ids[idx] for idx in sorted_corpus_indices]
            relevant_corpus_ids = relevance_dict[query_id]
            
            for k in k_values:
                top_k_corpus_ids = sorted_corpus_ids[:k]
                relevance_scores = [1 if cid in relevant_corpus_ids else 0 for cid in top_k_corpus_ids]
                ndcg_score = ndcg_at_k(relevance_scores, k)
                ndcg_scores_dict[k].append(ndcg_score)
    
    mean_ndcg_dict = {}
    for k in k_values:
        mean_ndcg = np.mean(ndcg_scores_dict[k])
        mean_ndcg_dict[k] = mean_ndcg
        print(f"NDCG@{k} for {dataset_name} ({split}): {mean_ndcg:.4f}")
    
    return mean_ndcg_dict


def main():
    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    print(f"Loading SparseEncoder model: {args.model_name}")
    model = SparseEncoder(args.model_name, trust_remote_code=True)
    print("Model loaded successfully")
    
    if args.topk_query is not None or args.topk_corpus is not None:
        print(f"\nTop-k Sparsification Settings:")
        if args.topk_query is not None:
            print(f"  Query top-k: {args.topk_query}")
        if args.topk_corpus is not None:
            print(f"  Corpus top-k: {args.topk_corpus}")
    
    model_name_clean = args.model_name.replace("/", "_")
    save_root_path = os.path.join(args.save_root_path, model_name_clean)
    os.makedirs(save_root_path, exist_ok=True)
    
    print(f"\n{'='*50}")
    print(f"Step 1: Generating embeddings for {args.dataset_name}")
    print(f"{'='*50}\n")
    get_retrieval_embeddings(
        model=model,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        save_root_path=save_root_path,
        topk_query=args.topk_query,
        topk_corpus=args.topk_corpus
    )
    
    # Step 2: 计算NDCG
    print(f"\n{'='*50}")
    print(f"Step 2: Calculating NDCG for {args.dataset_name}")
    print(f"{'='*50}\n")
    use_gpu = torch.cuda.is_available()
    ndcg_results = calculate_ndcg_from_embeddings(
        embeddings_path=save_root_path,
        dataset_name=args.dataset_name,
        split=args.split,
        k_values=args.k_values,
        use_gpu=use_gpu,
        batch_size=args.batch_size
    )
    
    print(f"\n{'='*50}")
    print(f"Final Results for {args.dataset_name}")
    print(f"{'='*50}")
    for k, score in ndcg_results.items():
        print(f"NDCG@{k}: {score:.4f}")
    
    results_file = os.path.join(save_root_path, f"{args.dataset_name}_results.json")
    with open(results_file, "w") as f:
        json.dump(ndcg_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()