"""把mteb和sentence transformer的数据集进行混合"""
from datasets import load_dataset, load_from_disk
from datasets import concatenate_datasets

max_pairs_per_dataset = 20000

dataset_1 = load_from_disk(f"./sentence_transformers_processed_datasets/combined_dataset_max_pairs_per_dataset_{max_pairs_per_dataset}")
dataset_2 = load_from_disk(f"./mteb_processed_datasets/combined_dataset_max_pairs_per_dataset_{max_pairs_per_dataset}")

for split in dataset_2.keys():
    dataset_2[split] = dataset_2[split].rename_columns({
        "sentence1": "anchor",
        "sentence2": "positive",
    })

train_merged = concatenate_datasets([dataset_1["train"], dataset_2["train"]]).shuffle(42)
test_merged = concatenate_datasets([dataset_1["test"], dataset_2["test"]]).shuffle(42)

from datasets import DatasetDict
merged_dataset = DatasetDict({
    "train": train_merged,
    "test": test_merged,
})

merged_dataset.save_to_disk("./dataset_for_mrl_finetuning_tiny")