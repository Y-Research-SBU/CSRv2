import pandas as pd
import os
from typing import List, Tuple
from datasets import Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm
import random
import argparse

parser = argparse.ArgumentParser(description="""
Process datasets and convert to sentence pairs for sentence transformer training.

This script processes various types of datasets (Pairs, Triplets, Sets, Query-Pairs, Query-Triplets)
and converts them into sentence pairs. It supports limiting the number of pairs extracted from each
dataset to balance the data distribution across different semantic domains.

Usage examples:
  python test.py                                          # Process all pairs from all datasets
  python test.py --max_pairs_per_dataset 10000           # Limit each dataset to 10,000 pairs
  python test.py --max_pairs_per_dataset 5000            # Limit each dataset to 5,000 pairs
""", formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("--max_pairs_per_dataset", type=int, required=True, 
                    help="""Maximum number of sentence pairs to extract from each dataset. 
                    If a dataset has more pairs than this limit, it will be shuffled and 
                    then sampled to this number. If a dataset has fewer pairs, all pairs 
                    will be kept. If not specified, all pairs from all datasets will be used.""")
args = parser.parse_args()

DATA_DIR = "./embedding-training-data"
MAX_ERRORS_TO_SHOW = 5 

def _is_str(x) -> bool:
    return isinstance(x, str)

def _is_list_of_str(x) -> bool:
    if not isinstance(x, list):
        return False
    return all(isinstance(v, str) for v in x)

def _validate_pairs_df(df: pd.DataFrame) -> Tuple[bool, List[Tuple[int, str]]]:
    errors = []
    if df.shape[1] != 2:
        return False, [(-1, f"Expected exactly 2 columns for Pairs, got {df.shape[1]}")]
    for idx, row in df.iterrows():
        v1, v2 = row.iloc[0], row.iloc[1]
        if not (_is_str(v1) and _is_str(v2)):
            errors.append((idx, f"Row {idx}: values must be strings; got types ({type(v1).__name__}, {type(v2).__name__})"))
            if len(errors) >= MAX_ERRORS_TO_SHOW:
                break
    return (len(errors) == 0), errors

def _validate_triplets_df(df: pd.DataFrame) -> Tuple[bool, List[Tuple[int, str]]]:
    errors = []
    cols = set(df.columns.astype(str))
    if cols == {"anchor", "positive", "negative"}:
        triplet_cols = ["anchor", "positive", "negative"]
        for idx, row in df.iterrows():
            a, p, n = row["anchor"], row["positive"], row["negative"]
            if not (_is_str(a) and _is_str(p) and _is_str(n)):
                errors.append((idx, f"Row {idx}: anchor/positive/negative must be strings; got types ({type(a).__name__},{type(p).__name__},{type(n).__name__})"))
                if len(errors) >= MAX_ERRORS_TO_SHOW:
                    break
        return (len(errors) == 0), errors

    if df.shape[1] == 3:
        for idx, row in df.iterrows():
            a, p, n = row.iloc[0], row.iloc[1], row.iloc[2]
            if not (_is_str(a) and _is_str(p) and _is_str(n)):
                errors.append((idx, f"Row {idx}: all 3 values must be strings; got types ({type(a).__name__},{type(p).__name__},{type(n).__name__})"))
                if len(errors) >= MAX_ERRORS_TO_SHOW:
                    break
        return (len(errors) == 0), errors

    return False, [(-1, f"Expected 3 columns for Triplets or named columns anchor/positive/negative; got {df.shape[1]} columns named {list(df.columns)}")]

def _validate_sets_df(df: pd.DataFrame) -> Tuple[bool, List[Tuple[int, str]]]:
    errors = []
    if list(df.columns) != ["set"]:
        return False, [(-1, f"Expected single column ['set'], got {list(df.columns)}")]
    for idx, row in df.iterrows():
        s = row["set"]
        if not _is_list_of_str(s):
            errors.append((idx, f"Row {idx}: 'set' must be a list[str]; got {type(s).__name__}"))
        elif len(s) < 1:
            errors.append((idx, f"Row {idx}: 'set' should not be empty"))
        if len(errors) >= MAX_ERRORS_TO_SHOW:
            break
    return (len(errors) == 0), errors

def _validate_query_pairs_df(df: pd.DataFrame) -> Tuple[bool, List[Tuple[int, str]]]:
    errors = []
    if set(df.columns) != {"query", "pos"}:
        return False, [(-1, f"Expected columns {{'query','pos'}}, got {set(df.columns)}")]
    for idx, row in df.iterrows():
        q, pos = row["query"], row["pos"]
        if not _is_str(q):
            errors.append((idx, f"Row {idx}: 'query' must be str; got {type(q).__name__}"))
        if not _is_list_of_str(pos):
            errors.append((idx, f"Row {idx}: 'pos' must be list[str]; got {type(pos).__name__}"))
        elif len(pos) < 1:
            errors.append((idx, f"Row {idx}: 'pos' should not be empty"))
        if len(errors) >= MAX_ERRORS_TO_SHOW:
            break
    return (len(errors) == 0), errors

def _validate_query_triplets_df(df: pd.DataFrame) -> Tuple[bool, List[Tuple[int, str]]]:
    errors = []
    if set(df.columns) != {"query", "pos", "neg"}:
        return False, [(-1, f"Expected columns {{'query','pos','neg'}}, got {set(df.columns)}")]
    for idx, row in df.iterrows():
        q, pos, neg = row["query"], row["pos"], row["neg"]
        if not _is_str(q):
            errors.append((idx, f"Row {idx}: 'query' must be str; got {type(q).__name__}"))
        if not _is_list_of_str(pos):
            errors.append((idx, f"Row {idx}: 'pos' must be list[str]; got {type(pos).__name__}"))
        elif len(pos) < 1:
            errors.append((idx, f"Row {idx}: 'pos' should not be empty"))
        if not _is_list_of_str(neg):
            errors.append((idx, f"Row {idx}: 'neg' must be list[str]; got {type(neg).__name__}"))
        elif len(neg) < 1:
            errors.append((idx, f"Row {idx}: 'neg' should not be empty"))
        if len(errors) >= MAX_ERRORS_TO_SHOW:
            break
    return (len(errors) == 0), errors

def convert_to_sentence_pairs(df: pd.DataFrame, schema: str, file_name: str, max_pairs: int = None) -> List[Tuple[str, str]]:
    pairs = []
    
    if schema == "Pairs":
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {file_name} (Pairs)"):
            pairs.append((str(row.iloc[0]), str(row.iloc[1])))
    
    elif schema == "Triplets":
        if set(df.columns.astype(str)) == {"anchor", "positive", "negative"}:
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {file_name} (Triplets-named)"):
                pairs.append((str(row["anchor"]), str(row["positive"])))
        else:
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {file_name} (Triplets-indexed)"):
                pairs.append((str(row.iloc[0]), str(row.iloc[1])))
    
    elif schema == "Sets":
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {file_name} (Sets)"):
            sentence_list = row["set"]
            if len(sentence_list) >= 2:
                for i in range(len(sentence_list)):
                    for j in range(i+1, len(sentence_list)):
                        pairs.append((str(sentence_list[i]), str(sentence_list[j])))
    
    elif schema == "Query-Pairs":
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {file_name} (Query-Pairs)"):
            query = str(row["query"])
            pos_list = row["pos"]
            for pos in pos_list:
                pairs.append((query, str(pos)))
    
    elif schema == "Query-Triplets":
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {file_name} (Query-Triplets)"):
            query = str(row["query"])
            pos_list = row["pos"]
            for pos in pos_list:
                pairs.append((query, str(pos)))
    
    if max_pairs is not None and len(pairs) > max_pairs:
        print(f"  📊 Original pairs: {len(pairs)}, sampling {max_pairs} pairs")
        random.seed(42) 
        pairs = random.sample(pairs, max_pairs)
        print(f"  ✂️  Sampled {len(pairs)} pairs from {file_name}")
    elif max_pairs is not None:
        print(f"  📊 Dataset has {len(pairs)} pairs (≤ max_pairs={max_pairs}), keeping all")
    
    return pairs

def create_dataset_from_pairs(pairs: List[Tuple[str, str]], file_name: str) -> DatasetDict:
    if len(pairs) == 0:
        empty_data = {"anchor": [], "positive": []}
        return DatasetDict({
            "train": Dataset.from_dict(empty_data),
            "test": Dataset.from_dict(empty_data)
        })
    
    random.seed(42)
    random.shuffle(pairs)
    
    split_idx = int(len(pairs) * 0.8)
    train_pairs = pairs[:split_idx]
    test_pairs = pairs[split_idx:]
    
    train_data = {
        "anchor": [pair[0] for pair in train_pairs],
        "positive": [pair[1] for pair in train_pairs]
    }
    
    test_data = {
        "anchor": [pair[0] for pair in test_pairs],
        "positive": [pair[1] for pair in test_pairs]
    }
    
    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)
    
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    print(f"  📊 Created dataset: {len(train_pairs)} train pairs, {len(test_pairs)} test pairs")
    
    return dataset_dict

def detect_and_validate(df: pd.DataFrame):
    validators = [
        ("Query-Triplets", _validate_query_triplets_df),
        ("Query-Pairs", _validate_query_pairs_df),
        ("Sets", _validate_sets_df),
        ("Triplets", _validate_triplets_df),
        ("Pairs", _validate_pairs_df),
    ]
    for name, fn in validators:
        ok, errs = fn(df.copy())
        if ok:
            return True, name, []
    aggregated = []
    for name, fn in validators:
        ok, errs = fn(df.copy())
        if not ok and errs:
            aggregated.append((name, errs[:MAX_ERRORS_TO_SHOW]))
    return False, None, aggregated

def main():
    if args.max_pairs_per_dataset:
        print(f"🔧 Maximum pairs per dataset: {args.max_pairs_per_dataset}")
    else:
        print("🔧 No limit on pairs per dataset (using all available pairs)")
    
    output_dir = "./sentence_transformers_processed_datasets"
    os.makedirs(output_dir, exist_ok=True)
    
    processed_datasets = []
    any_files = False
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".jsonl.gz")]
    files.sort()
    
    for file in files:
        any_files = True
        print(f"\n🔄 Processing {file}")
        
        df = pd.read_json(
            f"{DATA_DIR}/{file}",
            lines=True,
            compression="gzip"
        )
        
        is_valid, schema, errors = detect_and_validate(df)
        
        if is_valid:
            print(f"  ✅ Detected schema: {schema}")
            pairs = convert_to_sentence_pairs(df, schema, file, args.max_pairs_per_dataset)
            if len(pairs) > 0:
                dataset_dict = create_dataset_from_pairs(pairs, file)
                processed_datasets.append({
                    # "name": dataset_name,
                    "dataset": dataset_dict,
                    # "path": save_path
                })
            else:
                print(f"  ⚠️  No valid pairs generated from {file}")
        else:
            print(f"  ❌ INVALID — does not match any known schema.")
            for cand_name, cand_errs in errors:
                print(f"    Tried {cand_name}, found issues:")
                for (idx, msg) in cand_errs:
                    print(f"      - {msg}")
    
    if not any_files:
        print(f"No .jsonl.gz files found in {DATA_DIR}")
        return
    
    if processed_datasets:
        print(f"\n🔄 Merging {len(processed_datasets)} datasets...")
        
        all_train_datasets = []
        all_test_datasets = []
        total_train_pairs = 0
        total_test_pairs = 0
        
        for dataset_info in processed_datasets:
            dataset = dataset_info["dataset"]
            all_train_datasets.append(dataset["train"])
            all_test_datasets.append(dataset["test"])
            total_train_pairs += len(dataset["train"])
            total_test_pairs += len(dataset["test"])
        
        if all_train_datasets:
            combined_train = concatenate_datasets(all_train_datasets)
            combined_test = concatenate_datasets(all_test_datasets)
            
            combined_dataset = DatasetDict({
                "train": combined_train,
                "test": combined_test
            })
            
            combined_path = os.path.join(output_dir, f"combined_dataset_max_pairs_per_dataset_{args.max_pairs_per_dataset}")
            combined_dataset.save_to_disk(combined_path)
            
            print(f"  📊 Combined dataset statistics:")
            print(f"    Train pairs: {len(combined_train)} (from {len(processed_datasets)} datasets)")
            print(f"    Test pairs: {len(combined_test)} (from {len(processed_datasets)} datasets)")
            if args.max_pairs_per_dataset:
                max_possible_total = len(processed_datasets) * args.max_pairs_per_dataset
                print(f"    Note: With max_pairs_per_dataset={args.max_pairs_per_dataset}, theoretical max would be {max_possible_total} total pairs")
            print(f"  💾 Saved combined dataset to: {combined_path}")
        else:
            print("  ⚠️  No datasets to combine")
    else:
        print("  ⚠️  No valid datasets processed")

if __name__ == "__main__":
    main()