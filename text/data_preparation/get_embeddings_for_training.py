import numpy as np
import argparse
import json
import os
from sklearn.model_selection import train_test_split


def combine_classification_embedding(task_list, embedding_path, save_root_path):
    combined_train_data = []
    combined_train_labels = []
    combined_test_data = []
    combined_test_labels = []
    current_label_offset = 0

    for task_name in task_list:
        print(f"Processing {task_name} dataset...")
        task_embedding_path = os.path.join(embedding_path, task_name)
        train_path = os.path.join(task_embedding_path, "train.npz")
        validation_path = os.path.join(task_embedding_path, "validation.npz")
        test_path = os.path.join(task_embedding_path, "test.npz")

        train_exists = os.path.exists(train_path)
        validation_exists = os.path.exists(validation_path)
        test_exists = os.path.exists(test_path)

        if train_exists or validation_exists:
            all_embeddings = []
            all_labels = []
            if train_exists:
                train_data = np.load(train_path)
                all_embeddings.append(train_data['data'])
                all_labels.append(train_data['label'])         
            if validation_exists:
                validation_data = np.load(validation_path)
                all_embeddings.append(validation_data['data'])
                all_labels.append(validation_data['label'])
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
        elif test_exists:
            # Only test set exists
            test_data = np.load(test_path)
            test_embeddings = test_data['data']
            test_labels = test_data['label']
            all_embeddings, _, all_labels, _ = train_test_split(
                test_embeddings, test_labels, train_size=0.8, random_state=42, stratify=test_labels
            )
        else:
            raise ValueError(f"Warning: No data files found for task {task_name}.")

        unique_labels = np.unique(all_labels)
        label_mapping = {}
        for i, original_label in enumerate(sorted(unique_labels)):
            label_mapping[original_label] = current_label_offset + i
        mapped_labels = np.array([label_mapping[label] for label in all_labels])

        if test_exists and (train_exists or validation_exists):
            combined_train_data.append(all_embeddings)
            combined_train_labels.append(mapped_labels)
            test_data = np.load(test_path)
            test_embeddings = test_data['data']
            test_labels = test_data['label']
            mapped_test_labels = np.array([label_mapping.get(label, -1) for label in test_labels])
            valid_mask = mapped_test_labels != -1
            if not np.any(valid_mask):
                raise ValueError(f"Warning: No valid test labels for {task_name}.")
            combined_test_data.append(test_embeddings[valid_mask])
            combined_test_labels.append(mapped_test_labels[valid_mask])
        elif test_exists:
            # only test set exists
            test_data = np.load(test_path)
            test_embeddings = test_data['data']
            test_labels = test_data['label']
            _, test_emb, _, test_lab = train_test_split(
                test_embeddings, test_labels, train_size=0.8, random_state=42, stratify=test_labels
            )
            mapped_test_labels = np.array([label_mapping[label] for label in test_lab])            
            combined_train_data.append(all_embeddings)
            combined_train_labels.append(mapped_labels)
            combined_test_data.append(test_emb)
            combined_test_labels.append(mapped_test_labels)
        current_label_offset += len(unique_labels)

    if combined_train_data:
        final_train_data = np.concatenate(combined_train_data, axis=0)
        final_train_labels = np.concatenate(combined_train_labels, axis=0)
        final_test_data = np.concatenate(combined_test_data, axis=0)
        final_test_labels = np.concatenate(combined_test_labels, axis=0)
        os.makedirs(save_root_path, exist_ok=True)
        np.savez(os.path.join(save_root_path, "train.npz"), data=final_train_data, label=final_train_labels)
        np.savez(os.path.join(save_root_path, "test.npz"), data=final_test_data, label=final_test_labels)
        total_classes = len(np.unique(np.concatenate([final_train_labels, final_test_labels])))
        print(f"Combined classification data saved to {save_root_path}")
        print(f"Train data shape: {final_train_data.shape}, Test data shape: {final_test_data.shape}")
        print(f"Total classes: {total_classes}")
    else:
        raise ValueError("No valid data found to combine.")

def combine_clustering_embedding(task_list, embedding_path, save_root_path):
    combined_train_data = []
    combined_train_labels = []
    combined_test_data = []
    combined_test_labels = []
    current_label_offset = 0

    for task_name in task_list:
        print(f"Processing {task_name} dataset...")
        task_embedding_path = os.path.join(embedding_path, task_name)
        train_path = os.path.join(task_embedding_path, "train.npz")
        validation_path = os.path.join(task_embedding_path, "validation.npz")
        test_path = os.path.join(task_embedding_path, "test.npz")
        train_exists = os.path.exists(train_path)
        validation_exists = os.path.exists(validation_path)
        test_exists = os.path.exists(test_path)

        if train_exists or validation_exists:
            all_embeddings = []
            all_labels = []
            if train_exists:
                train_data = np.load(train_path)
                all_embeddings.append(train_data['data'])
                all_labels.append(train_data['label'])         
            if validation_exists:
                validation_data = np.load(validation_path)
                all_embeddings.append(validation_data['data'])
                all_labels.append(validation_data['label'])
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
        elif test_exists:
            # only test set exists
            test_data = np.load(test_path)
            test_embeddings = test_data['data']
            test_labels = test_data['label']
            all_embeddings, _, all_labels, _ = train_test_split(
                test_embeddings, test_labels, train_size=0.8, random_state=42
            )
        else:
            raise ValueError(f"Warning: No data files found for task {task_name}.")

        unique_labels = np.unique(all_labels)
        label_mapping = {}
        for i, original_label in enumerate(sorted(unique_labels)):
            label_mapping[original_label] = current_label_offset + i
        mapped_labels = np.array([label_mapping[label] for label in all_labels])

        if test_exists and (train_exists or validation_exists):
            combined_train_data.append(all_embeddings)
            combined_train_labels.append(mapped_labels)
            test_data = np.load(test_path)
            test_embeddings = test_data['data']
            test_labels = test_data['label']
            mapped_test_labels = np.array([label_mapping.get(label, -1) for label in test_labels])
            valid_mask = mapped_test_labels != -1
            if not np.any(valid_mask):
                raise ValueError(f"Warning: No valid test labels for {task_name}, skipping test.")
            combined_test_data.append(test_embeddings[valid_mask])
            combined_test_labels.append(mapped_test_labels[valid_mask])
        elif test_exists:
            test_data = np.load(test_path)
            test_embeddings = test_data['data']
            test_labels = test_data['label']
            _, test_emb, _, test_lab = train_test_split(
                test_embeddings, test_labels, train_size=0.8, random_state=42
            )
            # Handle case where test_lab might contain labels not in label_mapping
            mapped_test_labels = np.array([label_mapping.get(label, -1) for label in test_lab])
            valid_mask = mapped_test_labels != -1
            if not np.any(valid_mask):
                print(f"Warning: No valid test labels for {task_name}, skipping test data.")
            else:
                combined_test_data.append(test_emb[valid_mask])
                combined_test_labels.append(mapped_test_labels[valid_mask])
            combined_train_data.append(all_embeddings)
            combined_train_labels.append(mapped_labels)
        current_label_offset += len(unique_labels)

    if combined_train_data:
        final_train_data = np.concatenate(combined_train_data, axis=0)
        final_train_labels = np.concatenate(combined_train_labels, axis=0)
        
        os.makedirs(save_root_path, exist_ok=True)
        np.savez(os.path.join(save_root_path, "train.npz"), data=final_train_data, label=final_train_labels)
        
        if combined_test_data:
            final_test_data = np.concatenate(combined_test_data, axis=0)
            final_test_labels = np.concatenate(combined_test_labels, axis=0)
            np.savez(os.path.join(save_root_path, "test.npz"), data=final_test_data, label=final_test_labels)
            total_classes = len(np.unique(np.concatenate([final_train_labels, final_test_labels])))
            print(f"Combined clustering data saved to {save_root_path}")
            print(f"Train data shape: {final_train_data.shape}, Test data shape: {final_test_data.shape}")
            print(f"Total classes: {total_classes}")
        else:
            total_classes = len(np.unique(final_train_labels))
            print(f"Combined clustering data saved to {save_root_path}")
            print(f"Train data shape: {final_train_data.shape}, No test data")
            print(f"Total classes: {total_classes}")
    else:
        raise ValueError("No valid data found to combine.")

def combine_retrieval_embedding(task_list, embedding_path, save_root_path):
    combined_train_data = []
    combined_train_labels = []
    combined_test_data = []
    combined_test_labels = []
    current_label_offset = 0

    for task_name in task_list:
        task_embedding_path = os.path.join(embedding_path, task_name)
        corpus_path = os.path.join(task_embedding_path, "corpus.npz")
        queries_path = os.path.join(task_embedding_path, "queries.npz")
        if not os.path.exists(corpus_path) or not os.path.exists(queries_path):
            raise ValueError(f"Missing corpus.npz or queries.npz for {task_name}, skipping...")
        corpus_data = np.load(corpus_path)
        queries_data = np.load(queries_path)        
        corpus_ids = corpus_data['id']
        corpus_embeddings = corpus_data['data']
        query_ids = queries_data['id']
        query_embeddings = queries_data['data']
        corpus_id_to_idx = {cid: idx for idx, cid in enumerate(corpus_ids)}
        query_id_to_idx = {qid: idx for idx, qid in enumerate(query_ids)}
        
        train_path = os.path.join(task_embedding_path, "train.npz")
        val_path = os.path.join(task_embedding_path, "val.npz")
        test_path = os.path.join(task_embedding_path, "test.npz")
        train_exists = os.path.exists(train_path)
        val_exists = os.path.exists(val_path)
        test_exists = os.path.exists(test_path)
        
        all_pairs = []
        train_pairs = []
        test_pairs = []
        if train_exists:
            train_split = np.load(train_path)
            train_pairs = list(zip(train_split['query_id'], train_split['corpus_id']))
            all_pairs.extend(train_pairs)            
        if val_exists:
            val_split = np.load(val_path)
            val_pairs = list(zip(val_split['query_id'], val_split['corpus_id']))
            train_pairs.extend(val_pairs)  # Merge val into train
            all_pairs.extend(val_pairs)
        if test_exists:
            test_split = np.load(test_path)
            test_pairs = list(zip(test_split['query_id'], test_split['corpus_id']))
            all_pairs.extend(test_pairs)
        
        if not train_exists and not val_exists and test_exists:
            unique_test_queries = list(set([qid for qid, _ in test_pairs]))
            np.random.seed(42)
            np.random.shuffle(unique_test_queries)
            split_idx = int(0.8 * len(unique_test_queries))
            train_query_set = set(unique_test_queries[:split_idx])
            train_pairs = [(qid, cid) for qid, cid in test_pairs if qid in train_query_set]
            test_pairs = [(qid, cid) for qid, cid in test_pairs if qid not in train_query_set]        
        query_to_corpus = {}
        for qid, cid in all_pairs:
            if qid not in query_to_corpus:
                query_to_corpus[qid] = []
            query_to_corpus[qid].append(cid)
        
        current_label = current_label_offset
        query_labels = {}
        corpus_labels = {}        
        for qid, corpus_list in query_to_corpus.items():
            query_labels[qid] = current_label
            for cid in corpus_list:
                corpus_labels[cid] = current_label
            current_label += 1
        
        paired_queries = set(query_to_corpus.keys())
        paired_corpus = set()
        for corpus_list in query_to_corpus.values():
            paired_corpus.update(corpus_list)    
        unpaired_queries = [qid for qid in query_ids if qid not in paired_queries]
        unpaired_corpus = [cid for cid in corpus_ids if cid not in paired_corpus]
        
        for qid in unpaired_queries:
            query_labels[qid] = current_label
            current_label += 1            
        for cid in unpaired_corpus:
            corpus_labels[cid] = current_label
            current_label += 1
        
        task_train_data = []
        task_train_labels = []
        for qid, cid in train_pairs:
            if qid in query_id_to_idx and cid in corpus_id_to_idx:
                task_train_data.append(query_embeddings[query_id_to_idx[qid]])
                task_train_labels.append(query_labels[qid])
                task_train_data.append(corpus_embeddings[corpus_id_to_idx[cid]])
                task_train_labels.append(corpus_labels[cid])
        for qid in unpaired_queries:
            if qid in query_id_to_idx:
                task_train_data.append(query_embeddings[query_id_to_idx[qid]])
                task_train_labels.append(query_labels[qid])
        for cid in unpaired_corpus:
            if cid in corpus_id_to_idx:
                task_train_data.append(corpus_embeddings[corpus_id_to_idx[cid]])
                task_train_labels.append(corpus_labels[cid])
        
        task_test_data = []
        task_test_labels = []        
        for qid, cid in test_pairs:
            if qid in query_id_to_idx and cid in corpus_id_to_idx:
                task_test_data.append(query_embeddings[query_id_to_idx[qid]])
                task_test_labels.append(query_labels[qid])
                task_test_data.append(corpus_embeddings[corpus_id_to_idx[cid]])
                task_test_labels.append(corpus_labels[cid])
        if task_train_data:
            combined_train_data.append(np.array(task_train_data))
            combined_train_labels.append(np.array(task_train_labels))
        if task_test_data:
            combined_test_data.append(np.array(task_test_data))
            combined_test_labels.append(np.array(task_test_labels))
        current_label_offset = current_label
        print(f"Task {task_name}: Train samples: {len(task_train_data)}, Test samples: {len(task_test_data)}")
    
    if combined_train_data:
        final_train_data = np.concatenate(combined_train_data, axis=0)
        final_train_labels = np.concatenate(combined_train_labels, axis=0)    
        os.makedirs(save_root_path, exist_ok=True)
        np.savez(os.path.join(save_root_path, "train.npz"), 
                data=final_train_data, label=final_train_labels)
        if combined_test_data:
            final_test_data = np.concatenate(combined_test_data, axis=0)
            final_test_labels = np.concatenate(combined_test_labels, axis=0)
            np.savez(os.path.join(save_root_path, "test.npz"), 
                    data=final_test_data, label=final_test_labels)
            total_classes = len(np.unique(np.concatenate([final_train_labels, final_test_labels])))
            print(f"Combined retrieval data saved to {save_root_path}")
            print(f"Train data shape: {final_train_data.shape}, Test data shape: {final_test_data.shape}")
            print(f"Total classes: {total_classes}")
        else:
            total_classes = len(np.unique(final_train_labels))
            print(f"Combined retrieval data saved to {save_root_path}")
            print(f"Train data shape: {final_train_data.shape}, No test data")
            print(f"Total classes: {total_classes}")
    else:
        raise ValueError("No valid retrieval data found to combine.")

def combine_sts_embedding(task_list, embedding_path, save_root_path, score_threshold=3.0):
    combined_train_data = []
    combined_train_labels = []
    combined_test_data = []
    combined_test_labels = []
    current_label_offset = 0

    for task_name in task_list:
        print(f"Processing {task_name} dataset...")
        task_embedding_path = os.path.join(embedding_path, task_name)
        train_path = os.path.join(task_embedding_path, "train.npz")
        validation_path = os.path.join(task_embedding_path, "validation.npz")
        test_path = os.path.join(task_embedding_path, "test.npz")
        train_exists = os.path.exists(train_path)
        validation_exists = os.path.exists(validation_path)
        test_exists = os.path.exists(test_path)
        if train_exists or validation_exists:
            all_embeddings = []
            all_labels = []
            current_label = 0    
            if train_exists:
                train_data = np.load(train_path)
                train_emb1 = train_data['sentence1']
                train_emb2 = train_data['sentence2']
                train_scores = train_data['score']
                for i in range(train_emb1.shape[0]):
                    if train_scores[i] >= score_threshold:
                        all_embeddings.append(train_emb1[i])
                        all_embeddings.append(train_emb2[i])
                        all_labels.append(current_label)
                        all_labels.append(current_label)
                        current_label += 1
                    else:
                        all_embeddings.append(train_emb1[i])
                        all_embeddings.append(train_emb2[i])
                        all_labels.append(current_label)
                        all_labels.append(current_label + 1)
                        current_label += 2
            if validation_exists:
                validation_data = np.load(validation_path)
                val_emb1 = validation_data['sentence1']
                val_emb2 = validation_data['sentence2']
                val_scores = validation_data['score']
                for i in range(val_emb1.shape[0]):
                    if val_scores[i] >= score_threshold:
                        all_embeddings.append(val_emb1[i])
                        all_embeddings.append(val_emb2[i])
                        all_labels.append(current_label)
                        all_labels.append(current_label)
                        current_label += 1
                    else:
                        all_embeddings.append(val_emb1[i])
                        all_embeddings.append(val_emb2[i])
                        all_labels.append(current_label)
                        all_labels.append(current_label + 1)  
                        current_label += 2                
            all_embeddings = np.array(all_embeddings)
            all_labels = np.array(all_labels)
        elif test_exists:
            test_data = np.load(test_path)
            test_emb1 = test_data['sentence1']
            test_emb2 = test_data['sentence2']
            test_scores = test_data['score']
            num_pairs = test_emb1.shape[0]
            train_pair_indices, test_pair_indices = train_test_split(
                range(num_pairs), train_size=0.8, random_state=42
            )
            train_embeddings = []
            train_labels = []
            current_label = 0
            for i in train_pair_indices:
                if test_scores[i] >= score_threshold:
                    train_embeddings.append(test_emb1[i])
                    train_embeddings.append(test_emb2[i])
                    train_labels.append(current_label)
                    train_labels.append(current_label)
                    current_label += 1
                else:
                    train_embeddings.append(test_emb1[i])
                    train_embeddings.append(test_emb2[i])
                    train_labels.append(current_label)
                    train_labels.append(current_label + 1)
                    current_label += 2
            all_embeddings = np.array(train_embeddings)
            all_labels = np.array(train_labels)
        else:
            raise ValueError(f"Warning: No data files found for task {task_name}.")

        unique_labels = np.unique(all_labels)
        label_mapping = {}
        for i, original_label in enumerate(sorted(unique_labels)):
            label_mapping[original_label] = current_label_offset + i
        mapped_labels = np.array([label_mapping[label] for label in all_labels])

        if test_exists and (train_exists or validation_exists):
            combined_train_data.append(all_embeddings)
            combined_train_labels.append(mapped_labels)
            test_data = np.load(test_path)
            test_emb1 = test_data['sentence1']
            test_emb2 = test_data['sentence2']
            test_scores = test_data['score']
            test_embeddings = []
            test_labels = []
            test_current_label = 0            
            for i in range(test_emb1.shape[0]):
                if test_scores[i] >= score_threshold:
                    test_embeddings.append(test_emb1[i])
                    test_embeddings.append(test_emb2[i])
                    test_labels.append(test_current_label)
                    test_labels.append(test_current_label)
                    test_current_label += 1
                else:
                    test_embeddings.append(test_emb1[i])
                    test_embeddings.append(test_emb2[i])
                    test_labels.append(test_current_label)
                    test_labels.append(test_current_label + 1)
                    test_current_label += 2
            test_embeddings = np.array(test_embeddings)
            test_labels = np.array(test_labels)
            unique_test_labels = np.unique(test_labels)
            test_label_mapping = {}
            for i, original_label in enumerate(sorted(unique_test_labels)):
                test_label_mapping[original_label] = current_label_offset + len(unique_labels) + i
            mapped_test_labels = np.array([test_label_mapping[label] for label in test_labels])
            combined_test_data.append(test_embeddings)
            combined_test_labels.append(mapped_test_labels)
            
        elif test_exists:
            mapped_train_labels = np.array([label_mapping[label] for label in all_labels])
            combined_train_data.append(all_embeddings)
            combined_train_labels.append(mapped_train_labels)
            test_data = np.load(test_path)
            test_emb1 = test_data['sentence1']
            test_emb2 = test_data['sentence2']
            test_scores = test_data['score']
            num_pairs = test_emb1.shape[0]
            _, test_pair_indices = train_test_split(
                range(num_pairs), train_size=0.8, random_state=42
            )
            test_embeddings = []
            test_labels = []
            test_current_label = 0
            for i in test_pair_indices:
                if test_scores[i] >= score_threshold:
                    test_embeddings.append(test_emb1[i])
                    test_embeddings.append(test_emb2[i])
                    test_labels.append(test_current_label)
                    test_labels.append(test_current_label)
                    test_current_label += 1
                else:
                    test_embeddings.append(test_emb1[i])
                    test_embeddings.append(test_emb2[i])
                    test_labels.append(test_current_label)
                    test_labels.append(test_current_label + 1)
                    test_current_label += 2            
            test_embeddings = np.array(test_embeddings)
            test_labels = np.array(test_labels)
            unique_test_labels = np.unique(test_labels)
            test_label_mapping = {}
            for i, original_label in enumerate(sorted(unique_test_labels)):
                test_label_mapping[original_label] = current_label_offset + len(unique_labels) + i
            mapped_test_labels = np.array([test_label_mapping[label] for label in test_labels])
            combined_test_data.append(test_embeddings)
            combined_test_labels.append(mapped_test_labels)
        
        if test_exists and (train_exists or validation_exists):
            current_label_offset += len(unique_labels) + len(unique_test_labels)
        elif test_exists:
            current_label_offset += len(unique_labels) + len(unique_test_labels)
        else:
            current_label_offset += len(unique_labels)

    if combined_train_data:
        final_train_data = np.concatenate(combined_train_data, axis=0)
        final_train_labels = np.concatenate(combined_train_labels, axis=0)
        final_test_data = np.concatenate(combined_test_data, axis=0)
        final_test_labels = np.concatenate(combined_test_labels, axis=0)
        os.makedirs(save_root_path, exist_ok=True)
        np.savez(os.path.join(save_root_path, "train.npz"), data=final_train_data, label=final_train_labels)
        np.savez(os.path.join(save_root_path, "test.npz"), data=final_test_data, label=final_test_labels)
        total_classes = len(np.unique(np.concatenate([final_train_labels, final_test_labels])))
        print(f"Combined STS data saved to {save_root_path}")
        print(f"Train data shape: {final_train_data.shape}, Test data shape: {final_test_data.shape}")
        print(f"Total classes: {total_classes}")
        print(f"Score threshold used: {score_threshold}")
    else:
        raise ValueError("No valid data found to combine.")


def combine_reranking_embedding(task_list, embedding_path, save_root_path):
    combined_train_data = []
    combined_train_labels = []
    combined_test_data = []
    combined_test_labels = []
    current_label_offset = 0

    for task_name in task_list:
        print(f"Processing {task_name} dataset...")
        task_embedding_path = os.path.join(embedding_path, task_name)
        train_path = os.path.join(task_embedding_path, "train.npz")
        validation_path = os.path.join(task_embedding_path, "validation.npz")
        test_path = os.path.join(task_embedding_path, "test.npz")

        train_exists = os.path.exists(train_path)
        validation_exists = os.path.exists(validation_path)
        test_exists = os.path.exists(test_path)

        def process_reranking_data(data_path):
            """Process a single reranking data file"""
            data = np.load(data_path, allow_pickle=True)
            queries = data['query']  # [num_of_query, embedding_dimension]
            positives = data['positive']  # [num_of_query, [num_of_positive_corpus, embedding_dimension]]
            negatives = data['negative']  # [num_of_query, [num_of_negative_corpus, embedding_dimension]]
            all_embeddings = []
            all_labels = []
            current_label = 0
            for i in range(len(queries)):
                query_emb = queries[i]
                pos_embs = positives[i]  # [num_of_positive_corpus, embedding_dimension]
                neg_embs = negatives[i]  # [num_of_negative_corpus, embedding_dimension]
                all_embeddings.append(query_emb)
                all_labels.append(current_label)
                if pos_embs is not None:
                    for pos_emb in pos_embs:
                        all_embeddings.append(pos_emb)
                        all_labels.append(current_label)
                current_label += 1
                if neg_embs is not None:
                    for neg_emb in neg_embs:
                        all_embeddings.append(neg_emb)
                        all_labels.append(current_label)
                        current_label += 1            
            return np.array(all_embeddings), np.array(all_labels)

        if train_exists or validation_exists:
            all_embeddings = []
            all_labels = []
            current_label = 0
            if train_exists:
                train_emb, train_lab = process_reranking_data(train_path)
                all_embeddings.append(train_emb)
                all_labels.append(train_lab + current_label)
                current_label += len(np.unique(train_lab))
            if validation_exists:
                val_emb, val_lab = process_reranking_data(validation_path)
                all_embeddings.append(val_emb)
                all_labels.append(val_lab + current_label)
                current_label += len(np.unique(val_lab))
            all_embeddings = np.concatenate(all_embeddings, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)

        elif test_exists:
            data = np.load(test_path, allow_pickle=True)
            queries = data['query']
            positives = data['positive']
            negatives = data['negative']            
            num_queries = len(queries)
            train_query_indices, test_query_indices = train_test_split(
                range(num_queries), train_size=0.8, random_state=42
            )
            train_embeddings = []
            train_labels = []
            current_label = 0
            for i in train_query_indices:
                query_emb = queries[i]
                pos_embs = positives[i]
                neg_embs = negatives[i]
                train_embeddings.append(query_emb)
                train_labels.append(current_label)
                if pos_embs is not None:
                    for pos_emb in pos_embs:
                        train_embeddings.append(pos_emb)
                        train_labels.append(current_label)
                current_label += 1
                if neg_embs is not None:
                    for neg_emb in neg_embs:
                        train_embeddings.append(neg_emb)
                        train_labels.append(current_label)
                        current_label += 1
            all_embeddings = np.array(train_embeddings)
            all_labels = np.array(train_labels)
        else:
            raise ValueError(f"Warning: No data files found for task {task_name}.")
        unique_labels = np.unique(all_labels)
        label_mapping = {}
        for i, original_label in enumerate(sorted(unique_labels)):
            label_mapping[original_label] = current_label_offset + i
        mapped_labels = np.array([label_mapping[label] for label in all_labels])
    
        if test_exists and (train_exists or validation_exists):
            combined_train_data.append(all_embeddings)
            combined_train_labels.append(mapped_labels)
            test_emb, test_lab = process_reranking_data(test_path)
            unique_test_labels = np.unique(test_lab)
            test_label_mapping = {}
            for i, original_label in enumerate(sorted(unique_test_labels)):
                test_label_mapping[original_label] = current_label_offset + len(unique_labels) + i                
            mapped_test_labels = np.array([test_label_mapping[label] for label in test_lab])
            combined_test_data.append(test_emb)
            combined_test_labels.append(mapped_test_labels)
        elif test_exists:
            mapped_train_labels = np.array([label_mapping[label] for label in all_labels])
            combined_train_data.append(all_embeddings)
            combined_train_labels.append(mapped_train_labels)
            data = np.load(test_path, allow_pickle=True)
            queries = data['query']
            positives = data['positive']
            negatives = data['negative']
            num_queries = len(queries)
            
            _, test_query_indices = train_test_split(
                range(num_queries), train_size=0.8, random_state=42
            )
            test_embeddings = []
            test_labels = []
            test_current_label = 0
            for i in test_query_indices:
                query_emb = queries[i]
                pos_embs = positives[i]
                neg_embs = negatives[i]
                test_embeddings.append(query_emb)
                test_labels.append(test_current_label)
                if pos_embs is not None:
                    for pos_emb in pos_embs:
                        test_embeddings.append(pos_emb)
                        test_labels.append(test_current_label)
                test_current_label += 1
                if neg_embs is not None:
                    for neg_emb in neg_embs:
                        test_embeddings.append(neg_emb)
                        test_labels.append(test_current_label)
                        test_current_label += 1
            test_embeddings = np.array(test_embeddings)
            test_labels = np.array(test_labels)
            unique_test_labels = np.unique(test_labels)
            test_label_mapping = {}
            for i, original_label in enumerate(sorted(unique_test_labels)):
                test_label_mapping[original_label] = current_label_offset + len(unique_labels) + i
            mapped_test_labels = np.array([test_label_mapping[label] for label in test_labels])
            combined_test_data.append(test_embeddings)
            combined_test_labels.append(mapped_test_labels)
        
        if test_exists and (train_exists or validation_exists):
            current_label_offset += len(unique_labels) + len(unique_test_labels)
        elif test_exists:
            current_label_offset += len(unique_labels) + len(unique_test_labels)
        else:
            current_label_offset += len(unique_labels)

    if combined_train_data:
        final_train_data = np.concatenate(combined_train_data, axis=0)
        final_train_labels = np.concatenate(combined_train_labels, axis=0)
        final_test_data = np.concatenate(combined_test_data, axis=0)
        final_test_labels = np.concatenate(combined_test_labels, axis=0)        
        os.makedirs(save_root_path, exist_ok=True)
        np.savez(os.path.join(save_root_path, "train.npz"), data=final_train_data, label=final_train_labels)
        np.savez(os.path.join(save_root_path, "test.npz"), data=final_test_data, label=final_test_labels)
        total_classes = len(np.unique(np.concatenate([final_train_labels, final_test_labels])))
        print(f"Combined reranking data saved to {save_root_path}")
        print(f"Train data shape: {final_train_data.shape}, Test data shape: {final_test_data.shape}")
        print(f"Total classes: {total_classes}")
    else:
        raise ValueError("No valid data found to combine.")

def combine_pair_classification_embedding(task_list, embedding_path, save_root_path):
    combined_train_data = []
    combined_train_labels = []
    combined_test_data = []
    combined_test_labels = []
    current_label_offset = 0

    for task_name in task_list:
        print(f"Processing {task_name} dataset...")
        task_embedding_path = os.path.join(embedding_path, task_name)
        train_path = os.path.join(task_embedding_path, "train.npz")
        validation_path = os.path.join(task_embedding_path, "validation.npz")
        test_path = os.path.join(task_embedding_path, "test.npz")
        train_exists = os.path.exists(train_path)
        validation_exists = os.path.exists(validation_path)
        test_exists = os.path.exists(test_path)

        if train_exists or validation_exists:
            all_embeddings = []
            all_labels = []
            current_label = current_label_offset
            if train_exists:
                train_data = np.load(train_path)
                train_emb1 = train_data['sentence1']
                train_emb2 = train_data['sentence2']
                train_pair_labels = train_data['label'].flatten()                
                for i in range(train_emb1.shape[0]):
                    if train_pair_labels[i] == 1:  
                        all_embeddings.append(train_emb1[i])
                        all_embeddings.append(train_emb2[i])
                        all_labels.append(current_label)
                        all_labels.append(current_label)
                        current_label += 1
                    else:    
                        all_embeddings.append(train_emb1[i])
                        all_embeddings.append(train_emb2[i])
                        all_labels.append(current_label)
                        all_labels.append(current_label + 1)
                        current_label += 2
            if validation_exists:
                validation_data = np.load(validation_path)
                val_emb1 = validation_data['sentence1']
                val_emb2 = validation_data['sentence2']
                val_pair_labels = validation_data['label'].flatten()                
                for i in range(val_emb1.shape[0]):
                    if val_pair_labels[i] == 1: 
                        all_embeddings.append(val_emb1[i])
                        all_embeddings.append(val_emb2[i])
                        all_labels.append(current_label)
                        all_labels.append(current_label)
                        current_label += 1
                    else: 
                        all_embeddings.append(val_emb1[i])
                        all_embeddings.append(val_emb2[i])
                        all_labels.append(current_label)
                        all_labels.append(current_label + 1)
                        current_label += 2
            all_embeddings = np.array(all_embeddings)
            all_labels = np.array(all_labels)

        elif test_exists:
            test_data = np.load(test_path)
            test_emb1 = test_data['sentence1']
            test_emb2 = test_data['sentence2']
            test_pair_labels = test_data['label'].flatten()
            num_pairs = test_emb1.shape[0]
            train_pair_indices, test_pair_indices = train_test_split(
                range(num_pairs), train_size=0.8, random_state=42, stratify=test_pair_labels
            )
            train_embeddings = []
            train_labels = []
            current_label = current_label_offset
            for i in train_pair_indices:
                if test_pair_labels[i] == 1: 
                    train_embeddings.append(test_emb1[i])
                    train_embeddings.append(test_emb2[i])
                    train_labels.append(current_label)
                    train_labels.append(current_label)
                    current_label += 1
                else: 
                    train_embeddings.append(test_emb1[i])
                    train_embeddings.append(test_emb2[i])
                    train_labels.append(current_label)
                    train_labels.append(current_label + 1)
                    current_label += 2
            all_embeddings = np.array(train_embeddings)
            all_labels = np.array(train_labels)
        else:
            raise ValueError(f"Warning: No data files found for task {task_name}.")

        unique_labels = np.unique(all_labels)
        label_mapping = {}
        for i, original_label in enumerate(sorted(unique_labels)):
            label_mapping[original_label] = current_label_offset + i

        mapped_labels = np.array([label_mapping[label] for label in all_labels])

        if test_exists and (train_exists or validation_exists):
            combined_train_data.append(all_embeddings)
            combined_train_labels.append(mapped_labels)
            test_data = np.load(test_path)
            test_emb1 = test_data['sentence1']
            test_emb2 = test_data['sentence2']
            test_pair_labels = test_data['label'].flatten()
            test_embeddings = []
            test_labels = []
            test_current_label = 0
            for i in range(test_emb1.shape[0]):
                if test_pair_labels[i] == 1: 
                    test_embeddings.append(test_emb1[i])
                    test_embeddings.append(test_emb2[i])
                    test_labels.append(test_current_label)
                    test_labels.append(test_current_label)
                    test_current_label += 1
                else:  
                    test_embeddings.append(test_emb1[i])
                    test_embeddings.append(test_emb2[i])
                    test_labels.append(test_current_label)
                    test_labels.append(test_current_label + 1)
                    test_current_label += 2
            test_embeddings = np.array(test_embeddings)
            test_labels = np.array(test_labels)
            unique_test_labels = np.unique(test_labels)
            test_label_mapping = {}
            for i, original_label in enumerate(sorted(unique_test_labels)):
                test_label_mapping[original_label] = current_label_offset + len(unique_labels) + i
            mapped_test_labels = np.array([test_label_mapping[label] for label in test_labels])
            combined_test_data.append(test_embeddings)
            combined_test_labels.append(mapped_test_labels)
            
        elif test_exists:
            test_embeddings = []
            test_labels = []
            test_current_label = 0
            for i in test_pair_indices:
                if test_pair_labels[i] == 1: 
                    test_embeddings.append(test_emb1[i])
                    test_embeddings.append(test_emb2[i])
                    test_labels.append(test_current_label)
                    test_labels.append(test_current_label)
                    test_current_label += 1
                else: 
                    test_embeddings.append(test_emb1[i])
                    test_embeddings.append(test_emb2[i])
                    test_labels.append(test_current_label)
                    test_labels.append(test_current_label + 1)
                    test_current_label += 2
            test_embeddings = np.array(test_embeddings)
            test_labels = np.array(test_labels)
            mapped_train_labels = np.array([label_mapping[label] for label in all_labels])
            combined_train_data.append(all_embeddings)
            combined_train_labels.append(mapped_train_labels)
            unique_test_labels = np.unique(test_labels)
            test_label_mapping = {}
            for i, original_label in enumerate(sorted(unique_test_labels)):
                test_label_mapping[original_label] = current_label_offset + len(unique_labels) + i
            mapped_test_labels = np.array([test_label_mapping[label] for label in test_labels])
            combined_test_data.append(test_embeddings)
            combined_test_labels.append(mapped_test_labels)
        
        if test_exists and (train_exists or validation_exists):
            current_label_offset += len(unique_labels) + len(unique_test_labels)
        elif test_exists:
            current_label_offset += len(unique_labels) + len(unique_test_labels)
        else:
            current_label_offset += len(unique_labels)

    if combined_train_data:
        final_train_data = np.concatenate(combined_train_data, axis=0)
        final_train_labels = np.concatenate(combined_train_labels, axis=0)
        final_test_data = np.concatenate(combined_test_data, axis=0)
        final_test_labels = np.concatenate(combined_test_labels, axis=0)
        os.makedirs(save_root_path, exist_ok=True)
        np.savez(os.path.join(save_root_path, "train.npz"), data=final_train_data, label=final_train_labels)
        np.savez(os.path.join(save_root_path, "test.npz"), data=final_test_data, label=final_test_labels)        
        total_classes = len(np.unique(np.concatenate([final_train_labels, final_test_labels])))
        print(f"Combined pair classification data saved to {save_root_path}")
        print(f"Train data shape: {final_train_data.shape}, Test data shape: {final_test_data.shape}")
        print(f"Total classes: {total_classes}")
    else:
        raise ValueError("No valid data found to combine.")

TaskType2Function = {
    "classification": combine_classification_embedding,
    "clustering": combine_clustering_embedding,
    "retrieval": combine_retrieval_embedding,
    "sts": combine_sts_embedding,
    "reranking": combine_reranking_embedding,
    "pair_classification": combine_pair_classification_embedding
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root_path", type=str, required=True, help="Root path to save the embeddings")
    parser.add_argument("--task_type", type=str, required=True, help="Task type for the evaluation", choices=[
        "classification", "retrieval", "clustering", "sts", "reranking", "pair_classification"
    ])
    parser.add_argument("--embedding_path", type=str, required=True, help="Path to the embedding files directory")
    args = parser.parse_args()

    with open("task_list.json", "r") as f:
        task_list = json.load(f)
    if args.task_type not in task_list:
        raise ValueError(f"Task type '{args.task_type}' not found in task_list.json")

    tasks_to_process = task_list[args.task_type]

    for task in tasks_to_process:
        if task not in os.listdir(args.embedding_path):
            print(f"Warning: Task '{task}' not found in {args.embedding_path}.")
            raise ValueError(f"Task '{task}' not found in {args.embedding_path}.")
    os.makedirs(args.save_root_path, exist_ok=True)
    save_root_path = os.path.join(args.save_root_path, args.task_type)
    if args.task_type in TaskType2Function:
        TaskType2Function[args.task_type](tasks_to_process, args.embedding_path, save_root_path)
    else:
        raise ValueError(f"Task type {args.task_type} not supported yet.")

if __name__ == "__main__":
    main()