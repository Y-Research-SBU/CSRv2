import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task_type", type=str, required=True, help="Task type for the evaluation", choices=[
    "classification", "retrieval", "clustering", "sts", "reranking", "pair_classification"
])
parser.add_argument("--model_name", type=str, required=True, help="Model name to use for embeddings")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size for processing")
parser.add_argument("--save_root_path", type=str, required=True, help="Root path to save embeddings")
parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to process")
parser.add_argument("--gpu", type=str, required=True, help="GPU to use for processing, e.g., '0' for the first GPU")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from torch.nn import functional as F
from tqdm import tqdm
import json
import pickle

with open("dataset_to_prompt.json", "r") as f:
    Dataset2Prompt = json.load(f)

def encode_and_normalize(model, texts, prompt):
    with torch.no_grad():
        embeddings = model.encode(texts, prompt=prompt)
        embeddings = torch.from_numpy(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

def process_texts_in_batches(model, texts, prompt, batch_size):
    embedding_buffer = []
    current_batch = []
    for text in tqdm(texts):
        current_batch.append(text)
        if len(current_batch) == batch_size:
            embeddings = encode_and_normalize(model, current_batch, prompt)
            embedding_buffer.append(embeddings)
            current_batch = []
    if current_batch:
        embeddings = encode_and_normalize(model, current_batch, prompt)
        embedding_buffer.append(embeddings)
    return np.vstack(embedding_buffer) if embedding_buffer else np.array([])

def setup_output_directory(save_root_path, dataset_name):
    output_dir = os.path.join(save_root_path, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def should_skip_split(output_dir, split):
    if os.path.exists(os.path.join(output_dir, f"{split}.npz")):
        print(f"Split {split} already processed, skipping...")
        return True
    return False

def process_paired_texts_in_batches(model, texts1, texts2, prompt, batch_size):
    embedding_buffer1 = []
    embedding_buffer2 = []
    current_batch1 = []
    current_batch2 = []
    for text1, text2 in tqdm(zip(texts1, texts2)):
        current_batch1.append(text1)
        current_batch2.append(text2)
        if len(current_batch1) == batch_size:
            embeddings1 = encode_and_normalize(model, current_batch1, prompt)
            embeddings2 = encode_and_normalize(model, current_batch2, prompt)
            embedding_buffer1.append(embeddings1)
            embedding_buffer2.append(embeddings2)
            current_batch1, current_batch2 = [], []
    if current_batch1:
        embeddings1 = encode_and_normalize(model, current_batch1, prompt)
        embeddings2 = encode_and_normalize(model, current_batch2, prompt)
        embedding_buffer1.append(embeddings1)
        embedding_buffer2.append(embeddings2)
    final_embeddings1 = np.vstack(embedding_buffer1) if embedding_buffer1 else np.array([])
    final_embeddings2 = np.vstack(embedding_buffer2) if embedding_buffer2 else np.array([])
    return final_embeddings1, final_embeddings2

def process_texts_with_ids_in_batches(model, texts, ids, prompt, batch_size):
    embedding_buffer = []
    current_batch_texts = []
    current_batch_ids = []
    id_buffer = []
    for text, text_id in tqdm(zip(texts, ids)):
        current_batch_texts.append(text)
        current_batch_ids.append(text_id)
        if len(current_batch_texts) == batch_size:
            embeddings = encode_and_normalize(model, current_batch_texts, prompt)
            embedding_buffer.append(embeddings)
            id_buffer.extend(current_batch_ids)
            current_batch_texts, current_batch_ids = [], []
    if current_batch_texts:
        embeddings = encode_and_normalize(model, current_batch_texts, prompt)
        embedding_buffer.append(embeddings)
        id_buffer.extend(current_batch_ids)
    final_embeddings = np.vstack(embedding_buffer) if embedding_buffer else np.array([])
    return final_embeddings, id_buffer

def get_classification_embeddings(model, prompt, dataset_name, batch_size, save_root_path):
        """
        Dataset structure:
        Includes train/val/test splits; each split provides at least "text" and "label" columns.
        Output layout:
        save_root_path/dataset_name contains files such as:
        train/val/test.npz with keys "data" and "label":
            1. data: [num_of_samples, embedding_dimension]
            2. label: [num_of_samples, 1]
        """
    assert dataset_name in [
        "amazon_massive_intent",
        "amazon_massive_scenario",
        "mtop_intent",
        "mtop_domain",
        "imdb",
        "tweet_sentiment_extraction",
        "emotion",
        "amazon_counterfactual",
        "toxic_conversations_50k",
        "banking77"
    ]
    try:
        dataset = load_dataset(f"mteb/{dataset_name}", "en")
    except:
        dataset = load_dataset(f"mteb/{dataset_name}")
    label_to_int_mapping = None
    sample_label = None
    for split in dataset.keys():
        if len(dataset[split]) > 0:
            sample_label = dataset[split][0]['label']
            break
    if sample_label is not None and isinstance(sample_label, str):
        print("Detected string labels, creating label-to-integer mapping...")
        all_unique_labels = set()
        # collect all unique labels
        for split in dataset.keys():
            for sample in dataset[split]:
                all_unique_labels.add(sample['label'])
        label_to_int_mapping = {label: idx for idx, label in enumerate(sorted(all_unique_labels))}
        # apply mapping to original labels
        for split in dataset.keys():
            def map_labels(example):
                example['label'] = label_to_int_mapping[example['label']]
                return example
            dataset[split] = dataset[split].map(map_labels)
        print("Successfully mapped string labels to integers in the original dataset")
    
    output_dir = setup_output_directory(save_root_path, dataset_name)
    
    for split in dataset.keys():
        if should_skip_split(output_dir, split):
            continue
        split_dataset = dataset[split]
        print(f"Processing {split} split with {len(split_dataset)} samples...")        
        texts = [line["text"] for line in tqdm(split_dataset)]
        labels = [line["label"] for line in split_dataset]
        final_embeddings = process_texts_in_batches(model, texts, prompt, batch_size)
        np.savez(os.path.join(output_dir, f"{split}.npz"),
                 data=final_embeddings, label=np.array(labels))
        print(f"Completed processing {split} split, saved {len(final_embeddings)} embeddings")

def get_retrieval_embeddings(model, prompt, dataset_name, batch_size, save_root_path):
    assert dataset_name in [
        "arguana",
        "ClimateFEVER_test_top_250_only_w_correct-v2",
        "cqadupstack-gaming",
        "cqadupstack-unix",
        "fiqa",  
        "nfcorpus",
        "scidocs",
        "scifact",
    ]
        """
        Dataset structure:
        train/val/test splits each provide at least "corpus-id" and "queries-id" columns.
        The corpus subset contains at least "_id", "title", and "text" columns.
        The queries subset contains at least "_id" and "text" columns.
        Output layout:
        save_root_path/dataset_name contains three files:
            1. corpus.npz with keys "id" and "data"
            2. queries.npz with keys "id" and "data"
            3. train/val/test.npz storing query_id and corpus_id pairs
        """
    dataset = load_dataset(f"mteb/{dataset_name}")
    output_dir = setup_output_directory(save_root_path, dataset_name)
    
    if os.path.exists(os.path.join(output_dir, "corpus.npz")):
        print(f"Corpus for dataset {dataset_name} already processed, skipping...")
    else:
        try:
            corpus_dataset = load_dataset("mteb/" + dataset_name, name="corpus")["corpus"]
        except:
            corpus_dataset = load_dataset("mteb/" + dataset_name, name="corpus")["test"]        
        print("Processing corpus...")
        corpus_texts = [line["title"] + line["text"] for line in tqdm(corpus_dataset)]
        corpus_ids = [line["_id"] for line in corpus_dataset]
        corpus_embeddings, corpus_id_buffer = process_texts_with_ids_in_batches(
            model, corpus_texts, corpus_ids, prompt, batch_size)
        np.savez(os.path.join(output_dir, "corpus.npz"), data=corpus_embeddings, id=corpus_id_buffer)
        print(f"Completed processing corpus, saved {len(corpus_embeddings)} embeddings")
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
    queries_embeddings, queries_id_buffer = process_texts_with_ids_in_batches(
        model, query_texts, query_ids, prompt, batch_size)
    np.savez(os.path.join(output_dir, "queries.npz"), data=queries_embeddings, id=queries_id_buffer)
    print(f"Completed processing queries, saved {len(queries_embeddings)} embeddings")

    for split in dataset.keys():
        split_dataset = dataset[split]
        query_id_buffer = [line["query-id"] for line in split_dataset]
        corpus_id_buffer = [line["corpus-id"] for line in split_dataset]
        
        np.savez(os.path.join(output_dir, f"{split}.npz"),
                 query_id=query_id_buffer, corpus_id=corpus_id_buffer)
        print(f"Completed processing {split} relations, saved {len(query_id_buffer)} pairs")

def get_clustering_embeddings(model, prompt, dataset_name, batch_size, save_root_path):
    assert dataset_name in [
        "arxiv-clustering-s2s", 
        "biorxiv-clustering-p2p", 
        "biorxiv-clustering-s2s",
        "twentynewsgroups-clustering", 
        "medrxiv-clustering-p2p",
        "medrxiv-clustering-s2s",
        "stackexchange-clustering",  
        "stackexchange-clustering-p2p",
    ]
        """
        Dataset structure:
        train/val/test splits each include "sentences" and "labels" columns.
        The sentences column holds sentence lists; labels holds the aligned label list.
        Output layout:
        save_root_path/dataset_name stores files such as train/val/test.npz with keys "data" and "label":
            1. data: [num_of_samples, embedding_dimension]
            2. label: [num_of_samples, 1]
        """
    dataset = load_dataset(f"mteb/{dataset_name}")
    output_dir = setup_output_directory(save_root_path, dataset_name)
    
    for split in dataset.keys():
        if should_skip_split(output_dir, split):
            continue
            
        processing_dataset = dataset[split]
        
        all_labels = set()
        for line in processing_dataset:
            if isinstance(line["labels"], list):
                for label in line["labels"]:
                    all_labels.add(label)
            else:
                all_labels.add(line["labels"])
        label2id = {label: idx for idx, label in enumerate(sorted(all_labels))}
        
        texts = []
        labels = []
        print(f"Processing {split} split...")
        for line in tqdm(processing_dataset):
            if isinstance(line["sentences"], list) and isinstance(line["labels"], list):
                sentence_list = line["sentences"]
                labels_list = line["labels"]
                for sentence, label in zip(sentence_list, labels_list):
                    texts.append(sentence)
                    labels.append(label2id[label])
            else:
                texts.append(line["sentences"])
                labels.append(label2id[line["labels"]])
        final_embeddings = process_texts_in_batches(model, texts, prompt, batch_size)
        
        np.savez(os.path.join(output_dir, f"{split}.npz"),
                 data=final_embeddings, label=np.array(labels))
        print(f"Completed processing {split} split, saved {len(final_embeddings)} embeddings")
        
def get_sts_embeddings(model, prompt, dataset_name, batch_size, save_root_path):
    assert dataset_name in ["sickr-sts",
                            "sts12-sts",
                            "sts13-sts",
                            "sts14-sts",
                            "sts15-sts",
                            "sts16-sts",
                            "stsbenchmark-sts",
                            "biosses-sts",
                            "sts17-crosslingual-sts",
                            "sts22-crosslingual-sts"]
    """
    Dataset structure:
    train/val/test splits each provide "sentence1", "sentence2", and "score" columns.
    Output layout:
    save_root_path/dataset_name stores files such as train/val/test.npz with keys "sentence1", "sentence2", and "score".
    """
    try:
        dataset = load_dataset("mteb/" + dataset_name, "en")
    except Exception:
        try:
            dataset = load_dataset("mteb/" + dataset_name, "en-en")
        except:
            dataset = load_dataset("mteb/" + dataset_name)
    
    output_dir = setup_output_directory(save_root_path, dataset_name)
    
    for split in dataset.keys():
        if should_skip_split(output_dir, split):
            continue
        processing_dataset = dataset[split]
        print(f"Processing {split} split...")
        sentence1_list = [line["sentence1"] for line in tqdm(processing_dataset)]
        sentence2_list = [line["sentence2"] for line in processing_dataset]
        scores = [line["score"] for line in processing_dataset]
        sentence1_embeddings, sentence2_embeddings = process_paired_texts_in_batches(
            model, sentence1_list, sentence2_list, prompt, batch_size)
        np.savez(os.path.join(output_dir, f"{split}.npz"),
                 sentence1=sentence1_embeddings, sentence2=sentence2_embeddings, score=scores)
        print(f"Completed processing {split} split, saved {len(sentence1_embeddings)} sentence pairs")
    
def get_reranking_embeddings(model, prompt, dataset_name, batch_size, save_root_path):
    assert dataset_name in ["stackoverflowdupquestions-reranking", 
                            "askubuntudupquestions-reranking", 
                            "scidocs-reranking"]
        """
        Dataset structure:
        train/test splits each provide at least "query", "positive", and "negative" columns.
        query is a string; positive and negative hold lists of text passages.
        Output layout:
        save_root_path/dataset_name stores train/val/test.npz files containing each query embedding and its positive/negative corpora:
            key "query": [num_queries, embedding_dimension]
            key "positive": [num_queries, [num_positive_corpus, embedding_dimension]]
            key "negative": [num_queries, [num_negative_corpus, embedding_dimension]]
        """
    dataset = load_dataset(f"mteb/{dataset_name}")
    for split in dataset.keys():
        if os.path.exists(os.path.join(save_root_path, dataset_name, f"{split}.npz")):
            print(f"Dataset {dataset_name} split {split} already processed, skipping...")
            continue
        processing_dataset = dataset[split]
        os.makedirs(os.path.join(save_root_path, dataset_name), exist_ok=True)
        query_embedding_buffer = []
        positive_embedding_buffer = []
        negative_embedding_buffer = []
        queries = []
        for line in tqdm(processing_dataset):
            queries.append(line["query"])            
            if len(queries) == batch_size:
                with torch.no_grad():
                    query_embeddings = model.encode(queries, prompt=prompt)
                    query_embeddings = torch.from_numpy(query_embeddings)
                    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
                    query_embedding_buffer.append(query_embeddings.cpu().numpy())
                    queries = []
        if queries:
            with torch.no_grad():
                query_embeddings = model.encode(queries, prompt=prompt)
                query_embeddings = torch.from_numpy(query_embeddings)
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
                query_embedding_buffer.append(query_embeddings.cpu().numpy())
        for line in tqdm(processing_dataset):
            positive_embedding_buffer_for_query = []
            negative_embedding_buffer_for_query = []
            positive_corpus = []
            negative_corpus = []
            if line["positive"]:
                for corpus in line["positive"]:
                    positive_corpus.append(corpus)
                    if len(positive_corpus) == batch_size:
                        with torch.no_grad():
                            positive_embeddings = model.encode(positive_corpus, prompt=prompt)
                            positive_embeddings = torch.from_numpy(positive_embeddings)
                            positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
                            positive_embedding_buffer_for_query.append(positive_embeddings.cpu().numpy())
                            positive_corpus = []
                if positive_corpus:
                    with torch.no_grad():
                        positive_embeddings = model.encode(positive_corpus, prompt=prompt)
                        positive_embeddings = torch.from_numpy(positive_embeddings)
                        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)
                        positive_embedding_buffer_for_query.append(positive_embeddings.cpu().numpy())
                        positive_corpus = []
            if positive_embedding_buffer_for_query:
                positive_embedding_buffer.append(np.vstack(positive_embedding_buffer_for_query))
            else:
                positive_embedding_buffer.append(None)
            if line["negative"]:
                for corpus in line["negative"]:
                    negative_corpus.append(corpus)
                    if len(negative_corpus) == batch_size:
                        with torch.no_grad():
                            negative_embeddings = model.encode(negative_corpus, prompt=prompt)
                            negative_embeddings = torch.from_numpy(negative_embeddings)
                            negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)
                            negative_embedding_buffer_for_query.append(negative_embeddings.cpu().numpy())
                            negative_corpus = []
                if negative_corpus:
                    with torch.no_grad():
                        negative_embeddings = model.encode(negative_corpus, prompt=prompt)
                        negative_embeddings = torch.from_numpy(negative_embeddings)
                        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)
                        negative_embedding_buffer_for_query.append(negative_embeddings.cpu().numpy())
                        negative_corpus = []
            
            if negative_embedding_buffer_for_query:
                negative_embedding_buffer.append(np.vstack(negative_embedding_buffer_for_query))
            else:
                negative_embedding_buffer.append(None)
        
        final_query_embeddings = np.vstack(query_embedding_buffer)
        positive_obj = np.array(positive_embedding_buffer, dtype=object)
        negative_obj = np.array(negative_embedding_buffer, dtype=object)
        np.savez(os.path.join(save_root_path, dataset_name, f"{split}.npz"),
                 query=final_query_embeddings,
                 positive=positive_obj,
                 negative=negative_obj)
        print(f"Completed processing {split} split, saved {len(final_query_embeddings)} query embeddings")

def get_pair_classification_embeddings(model, prompt, dataset_name, batch_size, save_root_path):
    assert dataset_name in ["twitterurlcorpus-pairclassification",
                            "sprintduplicatequestions-pairclassification"]
        """
        Dataset structure:
        train/val/test splits each include "sent1", "sent2", and "label" columns.
        Output layout:
        save_root_path/dataset_name contains train/val/test.npz with keys:
            1. sentence1: [num_pairs, embedding_dimension]
            2. sentence2: [num_pairs, embedding_dimension]
            3. label: [num_pairs, 1]
        """
    dataset = load_dataset(f"mteb/{dataset_name}")
    output_dir = setup_output_directory(save_root_path, dataset_name)
    
    for split in dataset.keys():
        if should_skip_split(output_dir, split):
            continue
            
        processing_dataset = dataset[split]        
        if "sentence1" in processing_dataset.column_names:
            processing_dataset = processing_dataset.rename_column("sentence1", "sent1")
        if "sentence2" in processing_dataset.column_names:
            processing_dataset = processing_dataset.rename_column("sentence2", "sent2")
        
        print(f"Processing {split} split...")
        sentence1_list = []
        sentence2_list = []
        labels = []
        for line in processing_dataset:
            for sentence1_text, sentence2_text, label in tqdm(zip(line["sent1"], line["sent2"], line["labels"])):
                sentence1_list.append(sentence1_text)
                sentence2_list.append(sentence2_text)
                labels.append(label)
        final_sentence1_embeddings, final_sentence2_embeddings = process_paired_texts_in_batches(
            model, sentence1_list, sentence2_list, prompt, batch_size)
        
        np.savez(os.path.join(output_dir, f"{split}.npz"),
                 sentence1=final_sentence1_embeddings,
                 sentence2=final_sentence2_embeddings,
                 label=np.array(labels))
        print(f"Completed processing {split} split, saved {len(final_sentence1_embeddings)} sentence pairs")

TaskType2Function = {
    "classification": get_classification_embeddings,
    "retrieval": get_retrieval_embeddings,
    "clustering": get_clustering_embeddings,
    "sts": get_sts_embeddings,
    "reranking": get_reranking_embeddings,
    "pair_classification": get_pair_classification_embeddings,
}
            
if __name__ == "__main__":
    model = SentenceTransformer(args.model_name, trust_remote_code=True)
    processing_function = TaskType2Function[args.task_type]
    prompt = Dataset2Prompt[args.dataset_name]
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    os.makedirs(args.save_root_path, exist_ok=True)
    save_root_path = os.path.join(args.save_root_path, args.model_name.replace("/", "_"))
    os.makedirs(save_root_path, exist_ok=True)
    processing_function(model=model,
                        prompt=prompt,
                        dataset_name=dataset_name,
                        batch_size=batch_size,
                        save_root_path=save_root_path)