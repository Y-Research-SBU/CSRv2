import os
import subprocess
import json
import argparse
import torch
import numpy as np
from datetime import datetime
from model_zoo import CSR, CustomDataset
from torch.utils.data import DataLoader
from get_personalized_results import calculate_accuracy_from_embeddings, calculate_ndcg_from_embeddings, \
    calculate_clustering_from_embeddings, calculate_sts_from_embeddings, calculate_pair_classification_from_embeddings, \
    calculate_reranking_from_embeddings

def parse_args():
    parser = argparse.ArgumentParser(description="CSR Pipeline - Train, Package and Evaluate Models")
    
    parser.add_argument("--eval_tasks", nargs="+", required=True,
                       help="List of tasks to evaluate (if not specified, uses --tasks)")
    
    parser.add_argument("--base_model", default="Qwen3-Embedding-4B", help="Base model name")
    parser.add_argument("--gpu", type=int, required=True, help="GPU device to use")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--embed_dim", required=True, type=int, help="Embedding dimension")
    parser.add_argument("--hidden_size", required=True, type=int, help="Hidden size")
    parser.add_argument("--topk", required=True, type=int, help="Top K parameter")
    parser.add_argument("--initial_topk", type=int, default=None)
    parser.add_argument("--k_decay_ratio", type=float, default=0.7, help="K decay ratio")
    parser.add_argument("--auxk", type=int, required=True, help="Auxiliary K parameter")
    parser.add_argument("--auxk_coef", type=float, required=True, help="Auxiliary K coefficient")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--dead_threshold", default=30, type=int, dest='dead_threshold', help='dead_threshold')

    parser.add_argument("--training_embedding_path", type=str, required=True, help="Path to embedding file for training")
    parser.add_argument("--training_script", default="./CSR_training.py", help="Path to training script")
    parser.add_argument("--eval_embedding_path", type=str, required=True, help="Path to embeddings for evaluation (embeddings by the backbone)")
    
    parser.add_argument("--use_label_CL", action="store_true", help="Use label-based contrastive learning")
    parser.add_argument("--cl_coef", type=float, default=0.1, help="Contrastive learning coefficient")

    parser.add_argument("--model_suffix", type=str, required=True, help="Unique suffix for model naming (e.g., test, v1, experimental)")

    return parser.parse_args()

DATASET_TO_TYPE = {}
with open("./task_list.json", "r") as f:
    task_list = json.load(f)
for task_type, tasks in task_list.items():
    for task in tasks:
        DATASET_TO_TYPE[task] = task_type

TaskType2Function = {
    "classification": calculate_accuracy_from_embeddings,
    "retrieval": calculate_ndcg_from_embeddings,
    "clustering": calculate_clustering_from_embeddings,
    "sts": calculate_sts_from_embeddings,
    "pair_classification": calculate_pair_classification_from_embeddings,
    "reranking": calculate_reranking_from_embeddings,
}

args = None

def log(msg):
    """简化的log函数，直接打印到终端"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)

def get_checkpoint_info():
    os.makedirs("./CSR_results", exist_ok=True)
    eval_results_root = "".join((
        f"./CSR_results/", 
        f"base_model_{args.base_model}-epochs_{args.epochs}-batch_size_{args.batch_size}-lr_{args.lr}-topk_{args.topk}-auxk_{args.auxk}-",
        f"auxk_coef_{args.auxk_coef}-cl_coef_{args.cl_coef}-embed_dim_{args.embed_dim}-hidden_size_{args.hidden_size}"
    ))
    if args.initial_topk is not None:
        eval_results_root += f"-initial_topk_{args.initial_topk}-k_decay_{args.k_decay_ratio}"
    if args.use_label_CL:
        eval_results_root += "-use_label_CL"
    if args.model_suffix:
        eval_results_root += f"-suffix_{args.model_suffix}"  
    checkpoint_info_file = os.path.join(eval_results_root, "trained_checkpoints.json")
    if os.path.exists(checkpoint_info_file):
        with open(checkpoint_info_file, 'r') as f:
            return json.load(f)
    else:
        log(f"[Warning] Checkpoint info file not found: {checkpoint_info_file}")
        return {}

def train_single_model():    
    file_name = "".join((
        f"CSR-base_model_{args.base_model}-epochs_{args.epochs}-batch_size_{args.batch_size}-lr_{args.lr}-topk_{args.topk}-auxk_{args.auxk}-",
        f"auxk_coef_{args.auxk_coef}-cl_coef_{args.cl_coef}-embed_dim_{args.embed_dim}-hidden_size_{args.hidden_size}"
    ))
    if args.initial_topk is not None:
        file_name += f"-initial_topk_{args.initial_topk}-k_decay_ratio_{args.k_decay_ratio}"
    if args.use_label_CL:
        file_name += "-use_label_CL"
    if args.model_suffix:
        file_name += f'-suffix_{args.model_suffix}'

    os.makedirs("ckpt", exist_ok=True)
    pth_path = os.path.join("ckpt", f"{file_name}.pth")
    
    train_args = [
        "python", args.training_script,
        "--epochs", str(args.epochs),
        "--base_model", args.base_model,
        "--pretrained_emb", args.training_embedding_path,
        "--gpu", "0",
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--topk", str(args.topk),
        "--auxk", str(args.auxk),
        "--auxk_coef", str(args.auxk_coef),
        "--cl_coef", str(args.cl_coef),
        "--embed_dim", str(args.embed_dim),
        "--hidden_size", str(args.hidden_size),
        "--model_suffix", args.model_suffix,
    ]
    
    if args.use_label_CL:
        train_args.append("--use_label_CL")
    
    if args.initial_topk is not None:
        train_args.extend([
            "--initial_topk", str(args.initial_topk),
            "--k_decay_ratio", str(args.k_decay_ratio),
        ])
    
    log(f"Executing command: {' '.join(train_args)}")
    
    try:
        process = subprocess.Popen(
            train_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        return_code = process.poll()
        if return_code != 0:
            log(f"[Error] Training failed with return code {return_code}!")
            return None
        else:
            log(f"Training done successfully")
            if os.path.exists(pth_path):
                checkpoint_info = {
                    'checkpoint_path': os.path.abspath(pth_path),
                    'file_name': file_name,
                    'embedding_path': args.training_embedding_path
                }
                log(f"Checkpoint saved: {pth_path}")
                return checkpoint_info
            else:
                log(f"[Warning] Expected checkpoint not found: {pth_path}")
                return None
    except subprocess.TimeoutExpired:
        log(f"[Error] Training timed out!")
        process.kill()
        return None
    except Exception as e:
        log(f"[Error] Exception occurred during training: {str(e)}")
        return None

def eval_all():
    log("Running evaluation on multiple tasks ...")
    checkpoint_info = get_checkpoint_info()
    if not checkpoint_info:
        log("[Warning] No trained checkpoint found for evaluation.")
        return
    log(f"Using trained model: {checkpoint_info['file_name']}")
    eval_tasks = args.eval_tasks
    log(f"Evaluating on tasks: {eval_tasks}")
    model = load_trained_model()
    if model is None:
        log("[Error] Failed to load trained model, cannot proceed with evaluation")
        return
    evaluation_results = {}
    for task in eval_tasks:
        try:
            result = evaluate_model_performance(model, task)
            evaluation_results[task] = result
            log(f"Evaluation completed for {task}")
        except Exception as e:
            log(f"[Error] Exception during evaluation of {task}: {str(e)}")
            evaluation_results[task] = {
                'task': task,
                'status': 'failed',
                'error': str(e)
            }
    if evaluation_results:
        eval_results_root = "".join((
            f"./CSR_results/", 
            f"base_model_{args.base_model}-epochs_{args.epochs}-batch_size_{args.batch_size}-lr_{args.lr}-topk_{args.topk}-auxk_{args.auxk}-",
            f"auxk_coef_{args.auxk_coef}-cl_coef_{args.cl_coef}-embed_dim_{args.embed_dim}-hidden_size_{args.hidden_size}"
        ))
        if args.initial_topk is not None:
            eval_results_root += f"-initial_topk_{args.initial_topk}-k_decay_{args.k_decay_ratio}"
        if args.use_label_CL:
            eval_results_root += "-label_CL"
        if args.model_suffix:
            eval_results_root += f"-suffix_{args.model_suffix}"
        results_file = os.path.join(eval_results_root, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        log(f"Evaluation results saved to: {results_file}")
    log("All evaluations completed")

def load_trained_model():
    checkpoint_info = get_checkpoint_info()
    if not checkpoint_info:
        log(f"[Error] No trained checkpoint found")
        return None
    checkpoint_path = checkpoint_info['checkpoint_path']
    try:
        model = CSR(n_latents=args.hidden_size, topk=args.topk, auxk=args.auxk, 
                   dead_threshold=args.dead_threshold, normalize=False, n_inputs=args.embed_dim)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        log(f"Successfully loaded model from {checkpoint_path}")
        return model
    except Exception as e:
        log(f"[Error] Failed to load model: {str(e)}")
        return None

def evaluate_model_performance(model, task_name):
    try:
        log(f"Running performance evaluation for {task_name}")
        task_type = DATASET_TO_TYPE[task_name]
        eval_results_root = "".join((
            f"./CSR_results_personalized_evaluation/", 
            f"base_model_{args.base_model}-epochs_{args.epochs}-batch_size_{args.batch_size}-lr_{args.lr}-topk_{args.topk}-auxk_{args.auxk}-",
            f"auxk_coef_{args.auxk_coef}-cl_coef_{args.cl_coef}-embed_dim_{args.embed_dim}-hidden_size_{args.hidden_size}"
        ))
        if args.initial_topk is not None:
            eval_results_root += f"-initial_topk_{args.initial_topk}-k_decay_{args.k_decay_ratio}"
        if args.use_label_CL:
            eval_results_root += "-label_CL"
        if args.model_suffix:
            eval_results_root += f"-suffix_{args.model_suffix}"
        calculation_function = TaskType2Function[task_type]
        evaluation_results = calculation_function(embeddings_path=args.eval_embedding_path,
                                                  dataset_name=task_name,
                                                  model=model)
        results = {
            'task': task_name,
            'task_type': task_type,
            'model_params': {
                'base_model': args.base_model,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'topk': args.topk,
                'auxk': args.auxk,
                'auxk_coef': args.auxk_coef,
                'cl_coef': args.cl_coef,
                'embed_dim': args.embed_dim,
                'hidden_size': args.hidden_size,
            },
            "main_score": evaluation_results
        }
        return results
        
    except Exception as e:
        log(f"[Error] Failed to evaluate model for {task_name}: {str(e)}")
        import traceback
        log(f"[Error] Traceback: {traceback.format_exc()}")
        return {
            'task': task_name,
            'status': 'failed',
            'error': str(e)
        }

def get_all_trained_models():
    """
    获取所有训练好的模型信息
    
    Returns:
        包含所有训练好的模型信息的字典
    """
    return get_checkpoint_info()
    

def main():
    global args
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    eval_results_root = "".join((
        f"./CSR_results/base_model_{args.base_model}-epochs_{args.epochs}-batch_size_{args.batch_size}-lr_{args.lr}-topk_{args.topk}-auxk_{args.auxk}-",
        f"auxk_coef_{args.auxk_coef}-cl_coef_{args.cl_coef}-embed_dim_{args.embed_dim}-hidden_size_{args.hidden_size}"
    ))
    if args.initial_topk is not None:
        eval_results_root += f"-initial_topk_{args.initial_topk}-k_decay_ratio_{args.k_decay_ratio}"
    if args.use_label_CL:
        eval_results_root += "-use_label_CL"
    if args.model_suffix:
        eval_results_root += f"-suffix_{args.model_suffix}" 
    os.makedirs(eval_results_root, exist_ok=True)
    os.makedirs("./log", exist_ok=True)
    
    log("CSR Pipeline Starting...")
    log(f"Model suffix: {args.model_suffix}")
    log(f"Training script: {args.training_script}")
    log(f"Training parameters: batch_size={args.batch_size}, lr={args.lr}")
    log(f"Model parameters: topk={args.topk}, auxk={args.auxk}, auxk_coef={args.auxk_coef}")
    log(f"Contrastive learning: cl_coef={args.cl_coef}, use_label_CL={args.use_label_CL}")
    log(f"Evaluation embeddings path: {args.eval_embedding_path}")
    
    eval_tasks = args.eval_tasks
    log(f"Tasks to evaluate: {eval_tasks}")
    
    log("Step 1: Train single model")
    checkpoint_info = train_single_model()
    
    if checkpoint_info:
        checkpoint_info_file = os.path.join(eval_results_root, "trained_checkpoints.json")
        with open(checkpoint_info_file, 'w') as f:
            json.dump(checkpoint_info, f, indent=2)
        log(f"Checkpoint information saved to: {checkpoint_info_file}")
        log("Successfully trained model:")
        log(f"  Checkpoint: {checkpoint_info['checkpoint_path']}")
        log(f"  Embedding: {checkpoint_info['embedding_path']}")
        log("Step 2: Evaluate model on multiple tasks")
        eval_all()
    else:
        log("[Error] Model training failed, skipping evaluation")
    
    log("All done.")

if __name__ == "__main__":
    main()