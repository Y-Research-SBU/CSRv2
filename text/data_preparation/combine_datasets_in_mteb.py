from datasets import load_from_disk, DatasetDict, concatenate_datasets
import os

if __name__ == "__main__":
    mteb_dataset_path = "./datasets_for_finetuning"
    max_rows_per_dataset = 20000
    train_datasets = []
    test_datasets = []

    for dataset_path in os.listdir(mteb_dataset_path):
        dataset = load_from_disk(os.path.join(mteb_dataset_path, dataset_path))

        if len(dataset) > max_rows_per_dataset:
            dataset = dataset.shuffle(seed=42).select(range(max_rows_per_dataset))

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

    dataset_combined = DatasetDict({
        "train": train_dataset_combined,
        "test": test_dataset_combined,
    })

    dataset_combined.save_to_disk(f"./mteb_processed_datasets/combined_dataset_max_pairs_per_dataset_{max_rows_per_dataset}")