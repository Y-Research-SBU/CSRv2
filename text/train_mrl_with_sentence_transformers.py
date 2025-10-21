import os
import argparse
import sys

def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0

def safe_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--load_from_disk",
        action="store_true",
        help="Load dataset from disk instead of downloading"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name(s) - can specify multiple datasets for joint training"
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen3-Embedding-4B",
        help="Model name or path to load"
    )
    
    parser.add_argument(
        "--loss",
        required=True,
        type=str,
        choices=["cosent", "multiple_negatives_ranking_loss"],
        help="Loss function to use for training"
    )
    
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device(s) to use (e.g., '0', '1,2', '0,1,2,3')"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per device batch size for training and evaluation"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
        help="Number of gradient accumulation steps"
    )
    
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length for input text"
    )
    
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank for training"
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha value for training"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--dataset_suffix",
        type=str,
        required=True,
    )
    
    return parser.parse_args()

args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
    util,
)
from sentence_transformers.losses import BatchAllTripletLoss, MultipleNegativesRankingLoss, CoSENTLoss, MatryoshkaLoss
from transformers import BitsAndBytesConfig
import torch
from peft import LoraConfig, TaskType
from datasets import load_dataset, load_from_disk
    
def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

def main():
    # args is already parsed globally and GPU environment is set
    
    # 根据是否是分布式训练来配置模型
    if is_distributed():
        # 分布式训练：不使用量化，让 DDP 处理设备分配
        model = SentenceTransformer(
            f"{args.model_name}",
            trust_remote_code=True,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                # "attn_implementation": "flash_attention_2",
            },
        )
    else:
        # 单卡训练：使用量化和自动设备映射
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = SentenceTransformer(
            f"Qwen/{args.model_name}",
            trust_remote_code=True,
            model_kwargs={
                "quantization_config": bnb_cfg,
                "torch_dtype": torch.bfloat16,
                # "attn_implementation": "flash_attention_2",
                "device_map": "auto",
            },
        )
    
    # 设置 tokenizer 的 padding_side 为 'left' 以兼容 Flash Attention
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        model.tokenizer.padding_side = 'left'
    elif hasattr(model, '_modules') and '0' in model._modules:
        # 对于 SentenceTransformer，tokenizer 通常在第一个模块中
        if hasattr(model._modules['0'], 'tokenizer'):
            model._modules['0'].tokenizer.padding_side = 'left'
    
    # 设置最大序列长度以减少内存使用
    model.max_seq_length = args.max_seq_length
    # model.gradient_checkpointing_enable()
    
    # LoRA 配置
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model.add_adapter(peft_config)
    
    # 处理数据集 - 支持单个或多个数据集
    if args.load_from_disk:
        dataset = load_from_disk(args.dataset)
    else:
        dataset = load_dataset(args.dataset)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if "test" in dataset else dataset["validation"]
        
    if args.loss == "cosent":
        loss = CoSENTLoss(model=model)
    else:
        loss = MultipleNegativesRankingLoss(
            model=model,
            scale=20.0, 
            similarity_fct=util.cos_sim 
        )
    
    loss = MatryoshkaLoss(
        model=model,
        loss=loss,
        matryoshka_dims=[8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    
    output_dir = f"output/{args.model_name}-{args.dataset_suffix}-alpha_{args.lora_alpha}-batch_size_{args.batch_size}-lr_{args.lr}"
    
    # 根据是否分布式训练调整参数
    if is_distributed():
        gradient_accumulation_steps = 1  # 减少梯度累积步数，因为有多个GPU
        report_to = "wandb" if is_main_process() else "none"  # 只在主进程报告到wandb
    else:
        gradient_accumulation_steps = args.gradient_accumulation_steps
        report_to = "wandb"
    
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=10,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,  # 增加logging频率以便观察
        save_steps=100,
        warmup_ratio=0.1,  # 增加warmup比例
        dataloader_pin_memory=False,
        save_strategy="steps",
        report_to=report_to,
        # 分布式训练相关参数
        ddp_find_unused_parameters=False,
        save_only_model=True,
        greater_is_better=False,
        logging_first_step=True,
    )
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,   
    )
    
    safe_print(f"Starting training for dataset(s): {args.dataset}")
    safe_print(f"Training dataset size: {len(train_dataset)}")
    safe_print(f"Evaluation dataset size: {len(eval_dataset)}")
    safe_print(f"Batch size: {args.batch_size}")
    safe_print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    safe_print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    safe_print(f"Max sequence length: {args.max_seq_length}")
    safe_print(f"LoRA r: {args.lora_r}, LoRA alpha: {args.lora_alpha}")
    safe_print(f"Learning rate: {args.lr}")
    
    trainer.train()
    
    # 只在主进程保存模型
    if is_main_process():
        save_path = f"./results/{args.model_name}-{args.dataset_suffix}-alpha_{args.lora_alpha}-batch_size_{args.batch_size}-lr_{args.lr}"
        model.save_pretrained(save_path)
        safe_print(f"Model saved to: {save_path}")
    
if __name__ == "__main__":
    main()