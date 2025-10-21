from datasets import load_dataset, Dataset, concatenate_datasets
import random
from collections import defaultdict, Counter
from tqdm import tqdm
import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task_type", type=str, required=True, 
                    choices=["classification", "clustering", "retrieval", "sts", "pair_classification", "reranking"])
parser.add_argument("--save_root", type=str, required=True)
parser.add_argument("--max_sample_per_dataset", type=int, default=20000)
args = parser.parse_args()
os.makedirs(args.save_root, exist_ok=True)

def process_single_classification_dataset(dataset_name, dataset_num):
    try:
        dataset = load_dataset("mteb/" + dataset_name)
    except:
        dataset = load_dataset("mteb/" + dataset_name, "en")
    if "train" in dataset.keys():
        train_dataset = dataset["train"]
        if "validation" in dataset.keys():
            train_dataset = concatenate_datasets([train_dataset, dataset["validation"]])
        test_dataset = dataset["test"]
    else:
        split_dataset = dataset["test"].train_test_split(test_size=0.2, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
        if "validation" in dataset.keys():
            train_dataset = concatenate_datasets([train_dataset, dataset["validation"]])
    label_counts = Counter(train_dataset["label"])
    total_samples = len(train_dataset)
    train_pairs_per_label = {}
    test_pairs_per_label = {}
    test_num = int(dataset_num * 0.2)
    for label, count in label_counts.items():
        proportion = count / total_samples
        train_pairs_per_label[label] = int(dataset_num * proportion)
        test_pairs_per_label[label] = int(test_num * proportion)
    train_texts_by_label = defaultdict(list)
    for text, label in zip(train_dataset["text"], train_dataset["label"]):
        train_texts_by_label[label].append(text)
    test_texts_by_label = defaultdict(list)
    for text, label in zip(test_dataset["text"], test_dataset["label"]):
        test_texts_by_label[label].append(text)
    
    def create_pairs(pairs_per_label, texts_by_label, seed_offset=0):
        sentence1_list = []
        sentence2_list = []
        random.seed(42 + seed_offset)
        for label, num_pairs in pairs_per_label.items():
            texts = texts_by_label[label]
            # Skip labels that have no texts in this dataset split
            if len(texts) == 0:
                continue
            if len(texts) < 2:
                texts = texts * 2
            sentences_needed = num_pairs * 2  
            times_per_sentence = max(1, sentences_needed // len(texts))
            expanded_texts = []
            for text in texts:
                expanded_texts.extend([text] * times_per_sentence)
            while len(expanded_texts) < sentences_needed:
                expanded_texts.append(random.choice(texts))
            if len(expanded_texts) > sentences_needed:
                expanded_texts = random.sample(expanded_texts, sentences_needed)
            random.shuffle(expanded_texts)
            for i in range(num_pairs):
                sentence1_list.append(expanded_texts[i * 2])
                sentence2_list.append(expanded_texts[i * 2 + 1])
        return sentence1_list, sentence2_list
    
    train_sentence1, train_sentence2 = create_pairs(train_pairs_per_label, train_texts_by_label, seed_offset=0)
    test_sentence1, test_sentence2 = create_pairs(test_pairs_per_label, test_texts_by_label, seed_offset=100)
    train_pairs_dataset = Dataset.from_dict({
        "sentence1": train_sentence1,
        "sentence2": train_sentence2
    })
    test_pairs_dataset = Dataset.from_dict({
        "sentence1": test_sentence1,
        "sentence2": test_sentence2
    })
    final_dataset = {
        "train": train_pairs_dataset,
        "test": test_pairs_dataset
    }
    return final_dataset

def process_single_clustering_dataset(dataset_name, dataset_num):
    dataset = load_dataset("mteb/" + dataset_name)
    if "train" in dataset.keys():
        train_dataset = dataset["train"]
        if "validation" in dataset.keys():
            train_dataset = concatenate_datasets([train_dataset, dataset["validation"]])
        test_dataset = dataset["test"]
    else:
        split_dataset = dataset["test"].train_test_split(test_size=0.2, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
        if "validation" in dataset.keys():
            train_dataset = concatenate_datasets([train_dataset, dataset["validation"]])
            
    def flatten_dataset(dataset):
        texts = []
        labels = []
        for row in dataset:
            if isinstance(row["sentences"], list) and isinstance(row["labels"], list):
                for sentence, label in zip(row["sentences"], row["labels"]):
                    texts.append(sentence)
                    labels.append(label)
            else:
                texts.append(row["sentences"])
                labels.append(row["labels"])
        return texts, labels
    
    train_texts, train_labels = flatten_dataset(train_dataset)
    test_texts, test_labels = flatten_dataset(test_dataset)

    label_counts = Counter(train_labels)
    total_samples = len(train_texts)    
    train_pairs_per_label = {}
    test_pairs_per_label = {}
    test_num = int(dataset_num * 0.2)
    for label, count in label_counts.items():
        proportion = count / total_samples
        train_pairs_per_label[label] = int(dataset_num * proportion)
        test_pairs_per_label[label] = int(test_num * proportion)
    train_texts_by_label = defaultdict(list)
    for text, label in zip(train_texts, train_labels):
        train_texts_by_label[label].append(text)
    test_texts_by_label = defaultdict(list)
    for text, label in zip(test_texts, test_labels):
        test_texts_by_label[label].append(text)
    
    def create_pairs(pairs_per_label, texts_by_label, seed_offset=0):
        sentence1_list = []
        sentence2_list = []
        random.seed(42 + seed_offset)
        for label, num_pairs in pairs_per_label.items():
            texts = texts_by_label[label]
            if len(texts) == 0:
                continue
            if len(texts) < 2:
                texts = texts * 2
            sentences_needed = num_pairs * 2  
            times_per_sentence = max(1, sentences_needed // len(texts))
            expanded_texts = []
            for text in texts:
                expanded_texts.extend([text] * times_per_sentence)
            while len(expanded_texts) < sentences_needed:
                expanded_texts.append(random.choice(texts))
            if len(expanded_texts) > sentences_needed:
                expanded_texts = random.sample(expanded_texts, sentences_needed)
            random.shuffle(expanded_texts)
            for i in range(num_pairs):
                sentence1_list.append(expanded_texts[i * 2])
                sentence2_list.append(expanded_texts[i * 2 + 1])
        return sentence1_list, sentence2_list
    
    train_sentence1, train_sentence2 = create_pairs(train_pairs_per_label, train_texts_by_label, seed_offset=0)
    test_sentence1, test_sentence2 = create_pairs(test_pairs_per_label, test_texts_by_label, seed_offset=100)
    train_pairs_dataset = Dataset.from_dict({
        "sentence1": train_sentence1,
        "sentence2": train_sentence2
    })
    test_pairs_dataset = Dataset.from_dict({
        "sentence1": test_sentence1,
        "sentence2": test_sentence2
    })
    final_dataset = {
        "train": train_pairs_dataset,
        "test": test_pairs_dataset
    }
    return final_dataset

def process_single_retrieval_dataset(dataset_name, dataset_num):
    try:
        corpus_dataset = load_dataset("mteb/" + dataset_name, name="corpus")["corpus"]
    except:
        corpus_dataset = load_dataset("mteb/" + dataset_name, name="corpus")["test"]
    try:
        query_dataset = load_dataset("mteb/" + dataset_name, name="queries")["queries"]
    except:
        try:
            query_dataset = load_dataset("mteb/" + dataset_name, name="queries")["train"]
        except:
            query_dataset = load_dataset("mteb/" + dataset_name, name="queries")["test"]
    if "train" in load_dataset("mteb/" + dataset_name).keys():
        train_dataset = load_dataset("mteb/" + dataset_name)["train"]
        if "validation" in load_dataset("mteb/" + dataset_name).keys():
            train_dataset = concatenate_datasets([train_dataset, load_dataset("mteb/" + dataset_name)["validation"]])
        test_dataset = load_dataset("mteb/" + dataset_name)["test"]
    else:
        split_dataset = load_dataset("mteb/" + dataset_name)["test"].train_test_split(test_size=0.2, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
        if "validation" in load_dataset("mteb/" + dataset_name).keys():
            train_dataset = concatenate_datasets([train_dataset, load_dataset("mteb/" + dataset_name)["validation"]])
    corpus_dict = {}
    for corpus in tqdm(corpus_dataset):
        corpus_dict[corpus["_id"]] = corpus["text"]
    query_dict = {}
    for query in query_dataset:
        query_dict[query["_id"]] = query["text"]
    train_pairs_dataset = {"sentence1": [], "sentence2": []}
    for relation in tqdm(train_dataset):
        if len(train_pairs_dataset["sentence1"]) >= dataset_num:
            break
        query_id = relation["query-id"]
        corpus_id = relation["corpus-id"]
        if query_id not in query_dict:
            continue
        query_text = query_dict[query_id]
        if corpus_id not in corpus_dict:
            continue
        corpus_text = corpus_dict[corpus_id]
        train_pairs_dataset["sentence1"].append(query_text)
        train_pairs_dataset["sentence2"].append(corpus_text)
    test_pairs_dataset = {"sentence1": [], "sentence2": []}
    for relation in tqdm(test_dataset):
        if len(test_pairs_dataset["sentence1"]) >= dataset_num:
            break
        query_id = relation["query-id"]
        corpus_id = relation["corpus-id"]
        query_text = query_dict[query_id]
        corpus_text = corpus_dict[corpus_id]
        test_pairs_dataset["sentence1"].append(query_text)
        test_pairs_dataset["sentence2"].append(corpus_text)
    train_pairs_dataset = Dataset.from_dict(train_pairs_dataset)
    test_pairs_dataset = Dataset.from_dict(test_pairs_dataset)
    final_dataset = {
        "train": train_pairs_dataset,
        "test": test_pairs_dataset
    }
    return final_dataset

def process_single_sts_dataset(dataset_name, dataset_num, score_threshold=1.0):
    keep_cols = ["sentence1", "sentence2", "score"]
    try:
        dataset = load_dataset("mteb/" + dataset_name, "en")
    except Exception:
        try:
            dataset = load_dataset("mteb/" + dataset_name, "en-en")
        except:
            dataset = load_dataset("mteb/" + dataset_name)
    for split in dataset.keys():
        cols_to_remove = [col for col in dataset[split].column_names if col not in keep_cols]
        dataset[split] = dataset[split].remove_columns(cols_to_remove)
        dataset[split] = dataset[split].select_columns(keep_cols)
        dataset[split] = dataset[split].filter(lambda example: example["score"] >= score_threshold)
    if "train" in dataset.keys():
        train_dataset = dataset["train"]
        if "validation" in dataset.keys():
            train_dataset = concatenate_datasets([train_dataset, dataset["validation"]])
        test_dataset = dataset["test"]
    else:
        split_dataset = dataset["test"].train_test_split(test_size=0.2, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
        if "validation" in dataset.keys():
            train_dataset = concatenate_datasets([train_dataset, dataset["validation"]])
    
    test_num = int(dataset_num * 0.2)
    if len(train_dataset) > dataset_num:
        train_dataset = train_dataset.shuffle(seed=42).select(range(dataset_num)) 
    if len(test_dataset) > test_num:
        test_dataset = test_dataset.shuffle(seed=42).select(range(test_num))
    
    final_dataset = {
        "train": train_dataset,
        "test": test_dataset
    }
    return final_dataset

def process_single_pair_classification_dataset(dataset_name, dataset_num):
    dataset = load_dataset("mteb/" + dataset_name)
    if "train" in dataset.keys():
        train_dataset = dataset["train"]
        if "validation" in dataset.keys():
            train_dataset = concatenate_datasets([train_dataset, dataset["validation"]])
        test_dataset = dataset["test"]
    else:
        train_dataset = dataset["test"]
        test_dataset = dataset["test"]
        if "validation" in dataset.keys():
            train_dataset = concatenate_datasets([train_dataset, dataset["validation"]])
    if "sentence1" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("sentence1", "sent1")
    if "sentence2" in train_dataset.column_names:
        train_dataset = train_dataset.rename_column("sentence2", "sent2")
    if "sentence1" in test_dataset.column_names:
        test_dataset = test_dataset.rename_column("sentence1", "sent1")
    if "sentence2" in test_dataset.column_names:
        test_dataset = test_dataset.rename_column("sentence2", "sent2")
    def create_positive_sentence_pairs(dataset, max_pairs=None):
        sent1_list = []
        sent2_list = []
        for line in dataset:
            sentence1s = line["sent1"]
            sentence2s = line["sent2"]
            labels = line["labels"]
            for index in tqdm(range(len(sentence1s))):
                if labels[index] == 1:
                    sent1_list.append(sentence1s[index])
                    sent2_list.append(sentence2s[index])
                    if max_pairs and len(sent1_list) >= max_pairs:
                        return sent1_list, sent2_list
        return sent1_list, sent2_list
    
    test_num = int(dataset_num * 0.2)
    
    train_sentence1, train_sentence2 = create_positive_sentence_pairs(train_dataset, max_pairs=dataset_num)
    test_sentence1, test_sentence2 = create_positive_sentence_pairs(test_dataset, max_pairs=test_num)
    
    if len(train_sentence1) < dataset_num and len(train_sentence1) > 0:
        random.seed(42)
        indices = list(range(len(train_sentence1)))
        while len(train_sentence1) < dataset_num:
            idx = random.choice(indices)
            train_sentence1.append(train_sentence1[idx])
            train_sentence2.append(train_sentence2[idx])
    
    if len(test_sentence1) < test_num and len(test_sentence1) > 0:
        random.seed(42)
        indices = list(range(len(test_sentence1)))
        while len(test_sentence1) < test_num:
            idx = random.choice(indices)
            test_sentence1.append(test_sentence1[idx])
            test_sentence2.append(test_sentence2[idx])
    
    train_pairs_dataset = Dataset.from_dict({
        "sentence1": train_sentence1,
        "sentence2": train_sentence2
    })    
    test_pairs_dataset = Dataset.from_dict({
        "sentence1": test_sentence1,
        "sentence2": test_sentence2
    })
    final_dataset = {
        "train": train_pairs_dataset,
        "test": test_pairs_dataset
    }
    return final_dataset    

def process_single_reranking_embeddings(dataset_name, dataset_num):
    dataset = load_dataset("mteb/" + dataset_name)
    if "train" in dataset.keys():
        train_dataset = dataset["train"]
        if "validation" in dataset.keys():
            train_dataset = concatenate_datasets([train_dataset, dataset["validation"]])
        test_dataset = dataset["test"]
    else:
        split_dataset = dataset["test"].train_test_split(test_size=0.2, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
        if "validation" in dataset.keys():
            train_dataset = concatenate_datasets([train_dataset, dataset["validation"]])
            
    def create_query_positive_pairs(dataset, max_pairs):
        pairs = []
        for row in tqdm(dataset):
            query = row["query"]
            positive_list = row["positive"]
            for positive_item in positive_list:
                pairs.append((query, positive_item))
        random.seed(42)
        random.shuffle(pairs) 
        sentence1_list, sentence2_list = zip(*pairs[:max_pairs])
        return list(sentence1_list), list(sentence2_list)
    
    test_num = int(dataset_num * 0.2)
    train_sentence1, train_sentence2 = create_query_positive_pairs(train_dataset, dataset_num)
    test_sentence1, test_sentence2 = create_query_positive_pairs(test_dataset, test_num)
    
    train_pairs_dataset = Dataset.from_dict({
        "sentence1": train_sentence1,
        "sentence2": train_sentence2
    })
    test_pairs_dataset = Dataset.from_dict({
        "sentence1": test_sentence1,
        "sentence2": test_sentence2
    })
    final_dataset = {
        "train": train_pairs_dataset,
        "test": test_pairs_dataset
    }
    
    return final_dataset

if __name__ == "__main__":
    with open("./task_list.json", "r") as f:
        task_list = json.load(f)
    datasets_to_process = task_list[args.task_type]
    TaskType2Function = {
        "classification": process_single_classification_dataset,
        "clustering": process_single_clustering_dataset,
        "retrieval": process_single_retrieval_dataset,
        "sts": process_single_sts_dataset,
        "pair_classification": process_single_pair_classification_dataset,
        "reranking": process_single_reranking_embeddings
    }
    for dataset_name in datasets_to_process:
        print(f"Processing dataset: {dataset_name}")
        final_dataset = TaskType2Function[args.task_type](dataset_name, args.max_sample_per_dataset)
        final_dataset["train"].save_to_disk(f"{args.save_root}/{dataset_name}_train")
        final_dataset["test"].save_to_disk(f"{args.save_root}/{dataset_name}_test")