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
        required=True,
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
    
    # Early stopping parameters
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=3,
        help="Number of evaluations with no improvement after which training will be stopped"
    )
    
    parser.add_argument(
        "--early_stopping_threshold",
        type=float,
        default=0.01,
        help="Minimum change in the monitored metric to qualify as an improvement"
    )
    
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        default="eval_loss",
        help="Metric to monitor for early stopping (default: eval_loss)"
    )
    
    parser.add_argument(
        "--disable_early_stopping",
        action="store_true",
        help="Disable early stopping mechanism"
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
    SparseEncoder
)
from sentence_transformers.losses import BatchAllTripletLoss, MultipleNegativesRankingLoss, CoSENTLoss
from transformers import BitsAndBytesConfig, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
import torch
from peft import LoraConfig, TaskType
from datasets import load_dataset, load_from_disk
import numpy as np
    
def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1

class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping callback that monitors a specified metric and stops training
    when no improvement is seen for a given number of evaluation steps.
    """
    
    def __init__(self, early_stopping_patience: int = 3, early_stopping_threshold: float = 0.0, 
                 early_stopping_metric: str = "eval_loss"):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_metric = early_stopping_metric
        
        # For loss metrics, smaller is better; for accuracy metrics, larger is better
        self.is_metric_better = self._determine_metric_direction()
        
        self.best_metric = None
        self.patience_counter = 0
        
    def _determine_metric_direction(self):
        """Determine if the metric should be minimized or maximized"""
        minimize_metrics = ['loss', 'perplexity', 'error']
        maximize_metrics = ['accuracy', 'f1', 'bleu', 'rouge']
        
        metric_lower = self.early_stopping_metric.lower()
        
        for minimize_metric in minimize_metrics:
            if minimize_metric in metric_lower:
                return lambda current, best: current < best - self.early_stopping_threshold
                
        for maximize_metric in maximize_metrics:
            if maximize_metric in metric_lower:
                return lambda current, best: current > best + self.early_stopping_threshold
                
        # Default: assume it's a loss metric (smaller is better)
        safe_print(f"Warning: Could not determine direction for metric '{self.early_stopping_metric}'. "
                   f"Assuming smaller is better (like loss).")
        return lambda current, best: current < best - self.early_stopping_threshold
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, 
                    model=None, logs=None, **kwargs):
        """Check if early stopping criteria are met after each evaluation"""
        if logs is None:
            return control
            
        current_metric = logs.get(self.early_stopping_metric)
        if current_metric is None:
            safe_print(f"Warning: Early stopping metric '{self.early_stopping_metric}' not found in logs. "
                       f"Available metrics: {list(logs.keys())}")
            return control
            
        if self.best_metric is None:
            self.best_metric = current_metric
            safe_print(f"Early stopping initialized. Monitoring '{self.early_stopping_metric}': {current_metric:.6f}")
            return control
            
        if self.is_metric_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.patience_counter = 0
            safe_print(f"Early stopping: New best {self.early_stopping_metric}: {current_metric:.6f}")
        else:
            self.patience_counter += 1
            safe_print(f"Early stopping: No improvement in {self.early_stopping_metric} "
                       f"({current_metric:.6f} vs best {self.best_metric:.6f}). "
                       f"Patience: {self.patience_counter}/{self.early_stopping_patience}")
            
            if self.patience_counter >= self.early_stopping_patience:
                safe_print(f"Early stopping triggered! No improvement in {self.early_stopping_metric} "
                           f"for {self.early_stopping_patience} consecutive evaluations.")
                safe_print(f"Best {self.early_stopping_metric}: {self.best_metric:.6f}")
                control.should_training_stop = True
                
        return control

def main():
    # args is already parsed globally and GPU environment is set
    
    if is_distributed():
        model = SparseEncoder(
            args.model_name,
            trust_remote_code=True,
            model_kwargs={
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2",
            },
        )
    else:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = SparseEncoder(
            args.model_name,
            trust_remote_code=True,
            model_kwargs={
                "quantization_config": bnb_cfg,
                "torch_dtype": torch.bfloat16,
                "attn_implementation": "flash_attention_2",
                "device_map": "auto",
            },
        )
    
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        model.tokenizer.padding_side = 'left'
    elif hasattr(model, '_modules') and '0' in model._modules:
        if hasattr(model._modules['0'], 'tokenizer'):
            model._modules['0'].tokenizer.padding_side = 'left'
    
    model.max_seq_length = args.max_seq_length
    model.gradient_checkpointing_enable()
    
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model.add_adapter(peft_config)
    
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

    output_dir = f"output/{args.model_name.replace('/', '-')}-{args.dataset_suffix}-alpha_{args.lora_alpha}-batch_size_{args.batch_size}-lr_{args.lr}"
    
    run_name = f"{args.model_name.replace('/', '-')}-{args.dataset_suffix}-r{args.lora_r}-alpha{args.lora_alpha}-bs{args.batch_size}-lr{args.lr}-{args.loss}"
    
    if is_distributed():
        gradient_accumulation_steps = 1
        report_to = "wandb" if is_main_process() else "none"
    else:
        gradient_accumulation_steps = args.gradient_accumulation_steps
        report_to = "wandb"
    
    # Determine metric for best model based on early stopping metric
    metric_for_best_model = args.early_stopping_metric if not args.disable_early_stopping else "eval_loss"
    
    # Determine if greater is better based on the metric
    greater_is_better = False  # Default for loss metrics
    if not args.disable_early_stopping:
        maximize_metrics = ['accuracy', 'f1', 'bleu', 'rouge']
        metric_lower = args.early_stopping_metric.lower()
        for maximize_metric in maximize_metrics:
            if maximize_metric in metric_lower:
                greater_is_better = True
                break
    
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=10,
        learning_rate=args.lr,
        gradient_accumulation_steps=gradient_accumulation_steps,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=200,
        eval_steps=100,
        warmup_ratio=0.1,
        dataloader_pin_memory=False,
        eval_strategy="steps",
        save_strategy="steps",
        report_to=report_to,
        run_name=run_name,
        ddp_find_unused_parameters=False,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        logging_first_step=True,
    )
    
    # Prepare callbacks list
    callbacks = []
    
    # Add early stopping callback if not disabled
    if not args.disable_early_stopping:
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold,
            early_stopping_metric=args.early_stopping_metric
        )
        callbacks.append(early_stopping_callback)
        safe_print(f"Early stopping enabled: patience={args.early_stopping_patience}, "
                   f"threshold={args.early_stopping_threshold}, metric={args.early_stopping_metric}")
    else:
        safe_print("Early stopping disabled")
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        callbacks=callbacks,   
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
    
    # Early stopping info
    if not args.disable_early_stopping:
        safe_print(f"Early stopping: patience={args.early_stopping_patience}, "
                   f"threshold={args.early_stopping_threshold}, metric={args.early_stopping_metric}")
    else:
        safe_print("Early stopping: DISABLED")
    
    trainer.train()
    
    if is_main_process():
        save_path = f"./results/{args.model_name.replace('/', '-')}-{args.dataset_suffix}-alpha_{args.lora_alpha}-batch_size_{args.batch_size}-lr_{args.lr}"
        model.save_pretrained(save_path)
        safe_print(f"Model saved to: {save_path}")
    
if __name__ == "__main__":
    main()