import os
import subprocess
import json
import shutil
import argparse
import torch
import numpy as np
from datetime import datetime
from model_zoo import CSR, CustomDataset
from torch.utils.data import DataLoader

with open("./MTEB_task_prompts.json") as f:
    model_prompts = json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="CSR Pipeline - Train, Package and Evaluate Models")
    
    parser.add_argument("--eval_tasks", nargs="+", required=True,
                       help="List of tasks to evaluate")
    
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
    parser.add_argument("--packaged_model_dir", type=str, required=True,
                       help="Specific directory path where the packaged model should be saved")
    
    parser.add_argument("--use_label_CL", action="store_true", help="Use label-based contrastive learning")
    parser.add_argument("--cl_coef", type=float, default=0.1, help="Contrastive learning coefficient")

    parser.add_argument("--model_suffix", type=str, required=True, help="Unique suffix for model naming (e.g., test, v1, experimental)")

    return parser.parse_args()

with open("./dataset_to_task.json") as f:
    TASK_TO_MTEB = json.load(f)

args = None

def log(msg):
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_msg = f"[{timestamp}] {msg}"
    print(log_msg)

def get_checkpoint_info():
    os.makedirs("./CSR_results_MTEB_evaluation", exist_ok=True)
    eval_results_root = "".join((
        f"./CSR_results_MTEB_evaluation/", 
        f"base_model_{args.base_model}-epochs_{args.epochs}-batch_size_{args.batch_size}-lr_{args.lr}-topk_{args.topk}-auxk_{args.auxk}-",
        f"auxk_coef_{args.auxk_coef}-cl_coef_{args.cl_coef}-embed_dim_{args.embed_dim}-hidden_size_{args.hidden_size}"
    ))
    if args.initial_topk is not None:
        eval_results_root += f"-initial_topk_{args.initial_topk}-k_decay_ratio_{args.k_decay_ratio}"
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
        "--dead_threshold", str(args.dead_threshold),
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

def package_models():
    log("Packaging models ...")
    
    checkpoint_info = get_checkpoint_info()
    if not checkpoint_info:
        raise ValueError("No trained checkpoint found for packaging.")
    
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
    
    safetensors_path = os.path.join("ckpt_sparse_encoder", f"{file_name}.safetensors")
    if not os.path.exists(safetensors_path):
        raise FileNotFoundError(f"Safetensors file not found: {safetensors_path}")
    try:
        log(f"Packaging model to: {args.packaged_model_dir}")
        target_dir = os.path.join(args.packaged_model_dir, "3_SparseAutoEncoder")
        os.makedirs(target_dir, exist_ok=True)
        target_model_path = os.path.join(target_dir, "model.safetensors")
        shutil.copy2(safetensors_path, target_model_path)
        log(f"Copied model to: {target_model_path}")
        
        config = {
            "input_dim": args.embed_dim,
            "hidden_dim": args.hidden_size,
            "k": args.topk,
            "k_aux": args.auxk,
            "normalize": False,
            "dead_threshold": args.dead_threshold
        }
        config_path = os.path.join(target_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        log(f"Created config.json: {config_path}")
        log("Successfully packaged model")
        return True
    except Exception as e:
        raise ValueError(f"[Error] Packaging failed: {str(e)}")

def eval_all():
    log("Running MTEB evaluation ...")
    from sentence_transformers import SparseEncoder
    import mteb

    eval_results_root = "".join((
        f"./CSR_results_MTEB_evaluation/base_model_{args.base_model}-epochs_{args.epochs}-batch_size_{args.batch_size}-lr_{args.lr}-topk_{args.topk}-auxk_{args.auxk}-",
        f"auxk_coef_{args.auxk_coef}-cl_coef_{args.cl_coef}-embed_dim_{args.embed_dim}-hidden_size_{args.hidden_size}"
    ))
    if args.initial_topk is not None:
        eval_results_root += f"-initial_topk_{args.initial_topk}-k_decay_ratio_{args.k_decay_ratio}"
    if args.use_label_CL:
        eval_results_root += "-use_label_CL"
    if args.model_suffix:
        eval_results_root += f"-suffix_{args.model_suffix}"    
    os.makedirs(eval_results_root, exist_ok=True)

    model_path = args.packaged_model_dir
    
    evaluation_tasks = []
    for task in args.eval_tasks:
        if task in TASK_TO_MTEB:
            mteb_task_name = TASK_TO_MTEB[task]
            evaluation_tasks.append(mteb_task_name)
        else:
            raise ValueError(f"Task '{task}' not found in TASK_TO_MTEB mapping.")
    
    log(f"Evaluating tasks: {args.eval_tasks}")
    log(f"MTEB task names: {evaluation_tasks}")

    try:
        print(f"Loading model from: {model_path}")        
        model = SparseEncoder(
            model_path, 
            tokenizer_kwargs={"padding_side": "left"},
        )
        model.prompts = model_prompts
        task_list = []
        for task in evaluation_tasks:
            try:
                task_list.append(mteb.get_task(task, hf_subsets=["en"]))
            except:
                task_list.append(mteb.get_task(task))
        evaluation = mteb.MTEB(tasks=task_list)
        evaluation.run(
            model,
            eval_splits=["test"],
            output_folder=eval_results_root,
            show_progress_bar=True,
            encode_kwargs={"convert_to_sparse_tensor": False, "batch_size": 2}
        )
        log(f"MTEB evaluation completed for all tasks, results at {eval_results_root}")
        
    except Exception as e:
        log(f"[Error] MTEB evaluation failed: {str(e)}")
        import traceback
        log(f"[Error] Traceback: {traceback.format_exc()}")
    
    log("All MTEB evaluations completed")

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

def get_all_trained_models():
    return get_checkpoint_info()
    

def main():
    global args
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    eval_results_root = "".join((
        f"./CSR_results_MTEB_evaluation/base_model_{args.base_model}-epochs_{args.epochs}-batch_size_{args.batch_size}-lr_{args.lr}-topk_{args.topk}-auxk_{args.auxk}-",
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
    
    log("CSR MTEB Evaluation Pipeline Starting...")
    log(f"Model suffix: {args.model_suffix}")
    log(f"Training script: {args.training_script}")
    log(f"Training parameters: batch_size={args.batch_size}, lr={args.lr}")
    log(f"Model parameters: topk={args.topk}, auxk={args.auxk}, auxk_coef={args.auxk_coef}")
    log(f"Contrastive learning: cl_coef={args.cl_coef}, use_label_CL={args.use_label_CL}")
    
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
        
        log("Step 2: Package models for MTEB evaluation")
        if package_models():
            log("Step 3: Run MTEB evaluation")
            eval_all()
        else:
            log("[Error] Model packaging failed, skipping MTEB evaluation")
    else:
        log("[Error] Model training failed, skipping packaging and evaluation")
    
    log("All done.")

if __name__ == "__main__":
    main()