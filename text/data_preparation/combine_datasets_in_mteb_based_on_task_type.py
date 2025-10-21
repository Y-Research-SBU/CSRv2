from datasets import load_from_disk, DatasetDict, concatenate_datasets
import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--task_type", type=str, required=True)
parser.add_argument("--max_rows_per_dataset", type=int, required=True)
parser.add_argument("--mteb_dataset_path", type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    mteb_dataset_path = args.mteb_dataset_path
    max_rows_per_dataset = args.max_rows_per_dataset
    train_datasets = []
    test_datasets = []

    with open("./task_list.json", "r") as f:
        tasks_map = json.load(f)

    task_list = tasks_map[args.task_type]

    for task_name in task_list:
        for dataset_path in os.listdir(mteb_dataset_path):
            if dataset_path.startswith(task_name):
                dataset = load_from_disk(os.path.join(mteb_dataset_path, dataset_path))
                if len(dataset) > max_rows_per_dataset and dataset_path.endswith("train"):
                    dataset = dataset.shuffle(seed=42).select(range(max_rows_per_dataset))
                if len(dataset) > max_rows_per_dataset * 0.3 and dataset_path.endswith("test"):
                    dataset = dataset.shuffle(seed=42).select(range(int(max_rows_per_dataset * 0.2)))
                if dataset_path.endswith("train"):
                    train_datasets.append(dataset)
                elif dataset_path.endswith("test"):
                    test_datasets.append(dataset)

    train_dataset_combined = concatenate_datasets(
        [ds.select_columns(["sentence1", "sentence2"]) for ds in train_datasets]
    )
    test_dataset_combined = concatenate_datasets(
        [ds.select_columns(["sentence1", "sentence2"]) for ds in test_datasets]
    )
    
    train_dataset_combined = train_dataset_combined.shuffle(seed=42)
    test_dataset_combined = test_dataset_combined.shuffle(seed=42)

    # Create a DatasetDict with train and test splits
    dataset_dict = DatasetDict({
        "train": train_dataset_combined,
        "test": test_dataset_combined
    })

    # Save as a single dataset with both splits
    os.makedirs(f"./MTEB_single_task_type_finetuning_dataset", exist_ok=True)
    dataset_dict.save_to_disk(f"./MTEB_single_task_type_finetuning_dataset/{args.task_type}")
    
    print(f"Dataset saved to: ./MTEB_single_task_type_finetuning_dataset/{args.task_type}")
    print(f"Train samples: {len(train_dataset_combined)}")
    print(f"Test samples: {len(test_dataset_combined)}")