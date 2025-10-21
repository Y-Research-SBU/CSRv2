import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--task_type", type=str, choices=[
        "classification", "retrieval", "clustering", "sts", "reranking", "pair_classification"
    ])
    parser.add_argument("--gpu", type=str, required=True)
    return parser.parse_args()

# 模块级别的导入

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, v_measure_score
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm
from model_zoo import CSR, CustomDataset
from torch.utils.data import TensorDataset, DataLoader


def dcg_at_k(relevance_scores, k):
    """
    计算DCG@k
    Args:
        relevance_scores: 相关性分数列表，按排序顺序
        k: 截断位置
    Returns:
        DCG@k值
    """
    relevance_scores = np.array(relevance_scores)[:k]
    gains = 2**relevance_scores - 1
    discounts = np.log2(np.arange(len(relevance_scores)) + 2)
    return np.sum(gains / discounts)

def ndcg_at_k(relevance_scores, k):
    """
    计算NDCG@k
    Args:
        relevance_scores: 相关性分数列表，按排序顺序
        k: 截断位置
    Returns:
        NDCG@k值
    """
    dcg = dcg_at_k(relevance_scores, k)
    # 计算理想DCG（IDCG）
    ideal_relevance_scores = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_relevance_scores, k)
    
    if idcg == 0:
        return 0.0
    return dcg / idcg

def calculate_ndcg_from_embeddings(model, embeddings_path, dataset_name, split="test", k_values=[1, 5, 10, 20, 30], use_gpu=True, batch_size=64):
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    if isinstance(k_values, int):
        k_values = [k_values]
    dataset_path = os.path.join(embeddings_path, dataset_name)
    corpus_data = np.load(os.path.join(dataset_path, "corpus.npz"))
    queries_data = np.load(os.path.join(dataset_path, "queries.npz"))
    relation_data = np.load(os.path.join(dataset_path, f"{split}.npz"))    
    corpus_embeddings = corpus_data["data"]
    corpus_ids = corpus_data["id"]
    queries_embeddings = queries_data["data"]
    queries_ids = queries_data["id"]
    query_ids_rel = relation_data["query_id"]
    corpus_ids_rel = relation_data["corpus_id"]
    corpus_latents = []
    num_corpus_batches = (len(corpus_embeddings) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(range(num_corpus_batches), desc="Processing corpus batches"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(corpus_embeddings))
            batch_embeddings = torch.from_numpy(corpus_embeddings[batch_start:batch_end]).float().to(device)
            _, _, latents, _, _ = model(batch_embeddings)
            corpus_latents.append(latents.cpu().numpy())
    corpus_latents = np.concatenate(corpus_latents, axis=0)
    queries_latents = []
    num_queries_batches = (len(queries_embeddings) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(range(num_queries_batches), desc="Processing queries batches"):
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(queries_embeddings))
            batch_embeddings = torch.from_numpy(queries_embeddings[batch_start:batch_end]).float().to(device)
            _, _, latents, _, _ = model(batch_embeddings)
            queries_latents.append(latents.cpu().numpy())
    queries_latents = np.concatenate(queries_latents, axis=0)
    corpus_embeddings_tensor = torch.from_numpy(corpus_latents).float().to(device)
    queries_embeddings_tensor = torch.from_numpy(queries_latents).float().to(device)
    corpus_embeddings_tensor = F.normalize(corpus_embeddings_tensor, p=2, dim=1)
    queries_embeddings_tensor = F.normalize(queries_embeddings_tensor, p=2, dim=1)
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
        similarities_batch = torch.mm(batch_queries_tensor, corpus_embeddings_tensor.t())  # [batch_size, num_corpus]
        sorted_indices_batch = torch.argsort(similarities_batch, dim=1, descending=True)  # [batch_size, num_corpus]
        sorted_indices_batch_cpu = sorted_indices_batch.cpu().numpy()
        for i, query_id in enumerate(batch_query_ids):
            sorted_indices = sorted_indices_batch_cpu[i]
            relevance_scores = []
            relevant_corpus_ids = relevance_dict[query_id]
            for idx in sorted_indices:
                corpus_id = corpus_ids[idx]
                score = 1 if corpus_id in relevant_corpus_ids else 0
                relevance_scores.append(score)
            for k in k_values:
                ndcg = ndcg_at_k(relevance_scores, k)
                ndcg_scores_dict[k].append(ndcg)
    mean_ndcg_dict = {}
    for k in k_values:
        mean_ndcg = np.mean(ndcg_scores_dict[k])
        mean_ndcg_dict[k] = mean_ndcg
        print(f"NDCG@{k} for {dataset_name} ({split}): {mean_ndcg:.4f}")
    return mean_ndcg_dict

def generate_embeddings_for_task(model, embedding_data_path, gpu=0, batch_size=128):
    """
    将backbone的embeddings输入到一个CSR model中, 以进行evaluation, 略过再生成一遍embeddings所需要浪费的大量时间
    """
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()           
    data = np.load(embedding_data_path)
    embeddings = data['data'] 
    labels = data.get('label', None)
    dataset_data = {'data': embeddings}
    if labels is not None:
        dataset_data['labels'] = labels 
    dataset = CustomDataset(dataset_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_latents = []
    all_labels = []
    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device)
            _, _, latents, _, _ = model(batch_data)            
            all_latents.append(latents.cpu().numpy())
            if labels is not None:
                all_labels.append(batch_labels.numpy())
        combined_latents = np.concatenate(all_latents, axis=0)
        if labels is not None:
            combined_labels = np.concatenate(all_labels, axis=0)
    return combined_latents, combined_labels

def calculate_accuracy_from_embeddings(model, embeddings_path, dataset_name, num_runs=1, random_state_base=42):
    """
    从保存的embeddings计算分类准确率
    
    使用train.npz中的数据训练逻辑回归分类器，在test.npz中的数据上测试
    
    Args:
        embeddings_path: embeddings保存的根路径
        dataset_name: 数据集名称
        num_runs: 运行次数（默认10次）
        random_state_base: 随机种子基数
    
    Returns:
        tuple: (每次运行的accuracy列表, 平均accuracy)
    """
    assert "train.npz" in os.listdir(os.path.join(embeddings_path, dataset_name)), \
        f"train.npz not found in {os.path.join(embeddings_path, dataset_name)}"
    assert "test.npz" in os.listdir(os.path.join(embeddings_path, dataset_name)), \
        f"test.npz not found in {os.path.join(embeddings_path, dataset_name)}"
    dataset_path = os.path.join(embeddings_path, dataset_name)

    train_embeddings, train_labels = generate_embeddings_for_task(model, os.path.join(dataset_path, "train.npz"))
    test_embeddings, test_labels = generate_embeddings_for_task(model, os.path.join(dataset_path, "test.npz"))

    if train_labels.ndim > 1:
        train_labels = train_labels.flatten()
    if test_labels.ndim > 1:
        test_labels = test_labels.flatten()
    
    accuracy_scores = []
        
    for run in tqdm(range(num_runs), desc="Classification runs"):
        random_state = random_state_base + run
        clf = LogisticRegression(
            random_state=random_state,
            max_iter=100,
            tol=1e-4,
            n_jobs=-1,    
        )
        clf.fit(train_embeddings, train_labels)
        y_pred = clf.predict(test_embeddings)
        accuracy = accuracy_score(test_labels, y_pred)
        accuracy_scores.append(accuracy)
        if run < 3:
            print(f"  Run {run+1}: Accuracy = {accuracy:.4f}")
    mean_accuracy = np.mean(accuracy_scores)
    print(f"\n=== Classification Results ===")
    print(f"Individual accuracies: {[f'{acc:.4f}' for acc in accuracy_scores]}")
    print(f"Mean accuracy: {mean_accuracy:.4f}")
    return {"accuracy_per_time": accuracy_scores, "mean_accuracy": mean_accuracy}

def calculate_clustering_from_embeddings(model, embeddings_path, dataset_name, split="test", num_runs=5, random_state_base=42, batch_size=32):
    """
    从保存的embeddings计算聚类的V-measure指标，使用Mini-batch K-means
    
    Args:
        embeddings_path: embeddings保存的根路径
        dataset_name: 数据集名称
        split: 数据分割名称（如"test", "train"等）
        num_runs: 运行次数（默认10次）
        random_state_base: 随机种子基数
        batch_size: Mini-batch K-means的batch size（默认32）
    
    Returns:
        tuple: (每次运行的v_measure列表, 平均v_measure)
    """
    dataset_path = os.path.join(embeddings_path, dataset_name)
    
    print(f"Loading embeddings and labels for {dataset_name} ({split})...")
    embeddings, labels = generate_embeddings_for_task(model, os.path.join(dataset_path, f"{split}.npz"), batch_size=batch_size)
    
    if labels.ndim > 1:
        labels = labels.flatten()
    
    n_clusters = len(np.unique(labels))
    v_measure_scores = []
    
    for run in tqdm(range(num_runs), desc="Clustering runs"):
        # 使用不同的随机种子来确保每次运行的随机性
        random_state = random_state_base + run
        
        # 创建并训练Mini-batch K-means聚类器
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size, 
            n_init="auto",  # 使用多次初始化以获得更稳定的结果
        )
        
        # 对embeddings进行聚类
        cluster_predictions = kmeans.fit_predict(embeddings)
        
        # 计算V-measure指标
        v_measure = v_measure_score(labels, cluster_predictions)
        v_measure_scores.append(v_measure)
        
        if run < 3:  # 只打印前3次的详细信息
            print(f"  Run {run+1}: V-measure = {v_measure:.4f}")
    
    # 计算平均V-measure和标准差
    mean_v_measure = np.mean(v_measure_scores)
    std_v_measure = np.std(v_measure_scores)
    
    print(f"\n=== Clustering Results ===")
    print(f"Individual V-measures: {[f'{vm:.4f}' for vm in v_measure_scores]}")
    print(f"Mean V-measure: {mean_v_measure:.4f} ± {std_v_measure:.4f}")
    print(f"Min V-measure: {min(v_measure_scores):.4f}")
    print(f"Max V-measure: {max(v_measure_scores):.4f}")

    return {"v_measure per time": v_measure_scores, "mean_v_measure": mean_v_measure}

def calculate_sts_from_embeddings(model, embeddings_path, dataset_name, split="test", use_gpu=True):
    from scipy.stats import spearmanr
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    dataset_path = os.path.join(embeddings_path, dataset_name)
    data_file = np.load(os.path.join(dataset_path, f"{split}.npz"))
    sentence1_embeddings = data_file["sentence1"] 
    sentence2_embeddings = data_file["sentence2"] 
    true_scores = data_file["score"]               
    if true_scores.ndim > 1:
        true_scores = true_scores.flatten()
    def generate_CSR_latent(model, embedding, batch_size=128):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        embedding = torch.from_numpy(embedding)
        dataset = TensorDataset(embedding)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_latents = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(device)
                _, _, latents, _, _ = model(batch)
                all_latents.append(latents.cpu().numpy())
            all_latents = np.concatenate(all_latents, axis=0)
        return all_latents
    sentence1_embeddings = generate_CSR_latent(model, sentence1_embeddings)
    sentence2_embeddings = generate_CSR_latent(model, sentence2_embeddings)
    sentence1_tensor = torch.from_numpy(sentence1_embeddings).float().to(device)
    sentence2_tensor = torch.from_numpy(sentence2_embeddings).float().to(device)
    sentence1_normalized = F.normalize(sentence1_tensor, p=2, dim=1)
    sentence2_normalized = F.normalize(sentence2_tensor, p=2, dim=1)
    cosine_similarities_tensor = torch.sum(sentence1_normalized * sentence2_normalized, dim=1)
    cosine_similarities = cosine_similarities_tensor.cpu().numpy()
    spearman_corr, p_value = spearmanr(cosine_similarities, true_scores)
    return {"spearman_corr": spearman_corr, "p_value": p_value}

def calculate_pair_classification_from_embeddings(model, embeddings_path, dataset_name, batch_size=128, split="test", use_gpu=True):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    data = np.load(os.path.join(embeddings_path, dataset_name, f"{split}.npz"))
    sentence1_embeddings = data["sentence1"]  # [num_pairs, embedding_dim]
    sentence2_embeddings = data["sentence2"]  # [num_pairs, embedding_dim] 
    labels = data["label"]  # [num_pairs, 1] or [num_pairs]
    if labels.ndim > 1:
        labels = labels.flatten()
    def generate_CSR_latent(model, embedding, batch_size=128):
        device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        embedding = torch.from_numpy(embedding)
        dataset = TensorDataset(embedding)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_latents = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch[0].to(device)
                _, _, latents, _, _ = model(batch)
                all_latents.append(latents.cpu().numpy())
            all_latents = np.concatenate(all_latents, axis=0)
        return all_latents
    num_pairs = len(labels)
    sentence1_embeddings = generate_CSR_latent(model, sentence1_embeddings, batch_size)
    sentence2_embeddings = generate_CSR_latent(model, sentence2_embeddings, batch_size)
    sentence1_tensor = torch.from_numpy(sentence1_embeddings).float().to(device)
    sentence2_tensor = torch.from_numpy(sentence2_embeddings).float().to(device)
    sentence1_normalized = F.normalize(sentence1_tensor, p=2, dim=1)
    sentence2_normalized = F.normalize(sentence2_tensor, p=2, dim=1)
    cosine_similarities = []
    num_batches = (num_pairs + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="Computing cosine similarity"):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_pairs)
        batch_s1 = sentence1_normalized[batch_start:batch_end]
        batch_s2 = sentence2_normalized[batch_start:batch_end]
        batch_cosine = torch.sum(batch_s1 * batch_s2, dim=1)
        cosine_similarities.append(batch_cosine.cpu().numpy())
    cosine_similarities = np.concatenate(cosine_similarities)
    ap_score = average_precision_score(labels, cosine_similarities)
    print(f"Dataset: {dataset_name}")
    print(f"Average Precision Score: {ap_score}")
    return {"ap_score": ap_score}

def calculate_average_precision(relevance_scores):
    relevance_scores = np.array(relevance_scores)
    relevant_positions = np.where(relevance_scores == 1)[0]
    if len(relevant_positions) == 0:
        return 0.0
    precisions = []
    for i, pos in enumerate(relevant_positions):
        precision_at_pos = (i + 1) / (pos + 1)
        precisions.append(precision_at_pos)    
    return np.mean(precisions)

def calculate_reranking_from_embeddings(model, embeddings_path, dataset_name, batch_size=8, split="test", use_gpu=True):
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = np.load(os.path.join(embeddings_path, dataset_name, f"{split}.npz"), allow_pickle=True)
    queries = data["query"]  # [num_queries, embedding_dim]
    positive_list = data["positive"]  # [num_queries, [num_positive, embedding_dim]]
    negative_list = data["negative"]  # [num_queries, [num_negative, embedding_dim]]
    queries_tensor = torch.from_numpy(queries).float().to(device)
    dataset = TensorDataset(queries_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_latents = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch[0].to(device)
            _, _, latents, _, _ = model(batch)
            all_latents.append(latents.cpu().numpy())
        all_latents = np.concatenate(all_latents, axis=0)
    queries_tensor = torch.from_numpy(all_latents).float().to(device)
    queries_tensor = F.normalize(queries_tensor, p=2, dim=1)  # L2归一化
    ap_scores = []
    skipped_queries = 0
    num_queries = len(queries)
    num_batches = (num_queries + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="Computing MAP batches"):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_queries)
        batch_queries_tensor = queries_tensor[batch_start:batch_end]  # [batch_size, embedding_dim]
        for i in range(batch_queries_tensor.size(0)):
            query_idx = batch_start + i
            query_tensor = batch_queries_tensor[i:i+1]  # [1, embedding_dim]
            positive_corpus = positive_list[query_idx]  # [num_positive, embedding_dim]
            negative_corpus = negative_list[query_idx]  # [num_negative, embedding_dim]
            
            # 处理None值的情况
            if positive_corpus is None:
                positive_corpus = np.array([])  # 创建空数组
            if negative_corpus is None:
                negative_corpus = np.array([])  # 创建空数组
            
            # 跳过没有任何corpus的query
            if len(positive_corpus) == 0 and len(negative_corpus) == 0:
                print(f"Warning: Query {query_idx} has no positive or negative corpus, skipping...")
                skipped_queries += 1
                continue
                
            # 处理没有negative corpus的情况
            if len(negative_corpus) == 0:
                # 如果没有negative corpus，只使用positive corpus
                all_corpus = positive_corpus
                relevance_labels = np.ones(len(positive_corpus))
            elif len(positive_corpus) == 0:
                # 如果没有positive corpus，只使用negative corpus
                all_corpus = negative_corpus
                relevance_labels = np.zeros(len(negative_corpus))
            else:
                # 正常情况：同时有positive和negative corpus
                all_corpus = np.vstack([positive_corpus, negative_corpus])  # [num_positive + num_negative, embedding_dim]
                relevance_labels = np.concatenate([
                    np.ones(len(positive_corpus)),
                    np.zeros(len(negative_corpus))
                ])
            corpus_tensor = torch.from_numpy(all_corpus).float().to(device)
            with torch.no_grad():
                _, _, corpus_tensor, _, _ = model(corpus_tensor)
            corpus_tensor = F.normalize(corpus_tensor, p=2, dim=1)  
            similarities = torch.mm(query_tensor, corpus_tensor.t()).squeeze(0)  # [num_corpus] 
            sorted_indices = torch.argsort(similarities, descending=True).cpu().numpy()
            sorted_relevance_scores = relevance_labels[sorted_indices]
            ap = calculate_average_precision(sorted_relevance_scores)
            ap_scores.append(ap)
    
    if len(ap_scores) == 0:
        print(f"Warning: No valid queries found for {dataset_name} ({split})")
        return {"map": 0.0}
    
    map_score = np.mean(ap_scores)
    if skipped_queries > 0:
        print(f"Note: Skipped {skipped_queries} queries with no corpus out of {num_queries} total queries")
    print(f"MAP for {dataset_name} ({split}): {map_score:.4f} (computed from {len(ap_scores)} queries)")
    return {"map": map_score}

TaskType2Function = {
    "classification": calculate_accuracy_from_embeddings,
    "retrieval": calculate_ndcg_from_embeddings,
    "clustering": calculate_clustering_from_embeddings,
    "sts": calculate_sts_from_embeddings,
    "pair_classification": calculate_pair_classification_from_embeddings,
    "reranking": calculate_reranking_from_embeddings,
}

if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    calculation_function = TaskType2Function[args.task_type]
    embeddings_path = args.embeddings_path
    dataset_name = args.dataset_name
    calculation_function(
        embeddings_path=embeddings_path,
        dataset_name=dataset_name,
    )
    
    