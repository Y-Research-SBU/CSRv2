import json
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Summarize CSR evaluation results")
    
    # 模型相关参数
    parser.add_argument("--base_model", default="Qwen3-Embedding-4B", help="Base model name")
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
    
    # 标签对比学习相关参数
    parser.add_argument("--use_label_CL", action="store_true", help="Use label-based contrastive learning")
    parser.add_argument("--cl_coef", type=float, default=0.1, help="Contrastive learning coefficient")
    
    # 模型命名相关参数  
    parser.add_argument("--model_suffix", type=str, required=True, help="Unique suffix for model naming")
    
    # 任务相关参数
    parser.add_argument("--task_type", type=str, required=True, help="Task type to evaluate")
    
    # 路径相关参数
    parser.add_argument("--result_root", type=str, default="./CSR_results", 
                       help="Root directory for results (default: ./CSR_results)")
    
    # 输出控制参数
    parser.add_argument("--verbose", action="store_true", 
                       help="Show detailed configuration information (default: show only average score and results file)")

    return parser.parse_args()

args = parse_args()

with open("task_list.json") as f:
    task_list = json.load(f)

results = {}

def get_classification_results(json_path, task_list):
    result_per_task = {}
    with open(json_path) as f:
        result_dict = json.load(f)
    for task in task_list:
        if task in result_dict.keys():
            result_per_task[task] = result_dict[task]["main_score"]["mean_accuracy"]
    average_score = sum(result_per_task.values()) / len(result_per_task) if result_per_task else 0
    return result_per_task, average_score

def get_clustering_results(json_path, task_list):
    result_per_task = {}
    with open(json_path) as f:
        result_dict = json.load(f)
    for task in task_list:
        if task in result_dict.keys():
            result_per_task[task] = result_dict[task]["main_score"]["mean_v_measure"]
    average_score = sum(result_per_task.values()) / len(result_per_task) if result_per_task else 0
    return result_per_task, average_score

def get_sts_results(json_path, task_list):
    result_per_task = {}
    with open(json_path) as f:
        result_dict = json.load(f)
    for task in task_list:
        if task in result_dict.keys():
            if "error" in result_dict[task]:
                return 0, 0
            if result_dict[task]["main_score"]["spearman_corr"] != "NaN":
                result_per_task[task] = result_dict[task]["main_score"]["spearman_corr"]
    average_score = sum(result_per_task.values()) / len(result_per_task) if result_per_task else 0
    return result_per_task, average_score

def get_retrieval_results(json_path, task_list):
    result_per_task = {}
    with open(json_path) as f:
        result_dict = json.load(f)
    have_error = False
    for task in task_list:
        if task in result_dict.keys():
            if "error" in result_dict[task].keys():
                have_error = True
                break
            result_per_task[task] = result_dict[task]["main_score"]["10"]
    if have_error:
        return 0, 0
    average_score = sum(result_per_task.values()) / len(result_per_task) if result_per_task else 0
    return result_per_task, average_score

def get_reranking_results(json_path, task_list):
    result_per_task = {}
    with open(json_path) as f:
        result_dict = json.load(f)
    have_error = False
    for task in task_list:
        if task in result_dict.keys():
            if "error" in result_dict[task].keys():
                have_error = True
                break
            result_per_task[task] = result_dict[task]["main_score"]["map"]
    if have_error:
        return 0, 0
    average_score = sum(result_per_task.values()) / len(result_per_task) if result_per_task else 0
    return result_per_task, average_score

def get_pair_classification_results(json_path, task_list):
    result_per_task = {}
    with open(json_path) as f:
        result_dict = json.load(f)
    have_error = False
    for task in task_list:
        if task in result_dict.keys():
            if "error" in result_dict[task].keys():
                have_error = True
                break
            result_per_task[task] = result_dict[task]["main_score"]["mean_accuracy"]
    if have_error:
        return 0, 0
    average_score = sum(result_per_task.values()) / len(result_per_task) if result_per_task else 0
    return result_per_task, average_score


task2function = {
    "classification": get_classification_results,
    "clustering": get_clustering_results,
    "retrieval": get_retrieval_results,
    "sts": get_sts_results, 
    "reranking": get_reranking_results,
    "pair_classification": get_pair_classification_results,
}

def build_result_path():
    """构建与训练代码一致的结果路径"""
    eval_results_root = "".join((
        f"{args.result_root}/", 
        f"base_model_{args.base_model}-epochs_{args.epochs}-batch_size_{args.batch_size}-lr_{args.lr}-topk_{args.topk}-auxk_{args.auxk}-",
        f"auxk_coef_{args.auxk_coef}-cl_coef_{args.cl_coef}-embed_dim_{args.embed_dim}-hidden_size_{args.hidden_size}"
    ))
    if args.initial_topk is not None:
        eval_results_root += f"-initial_topk_{args.initial_topk}-k_decay_{args.k_decay_ratio}"
    if args.use_label_CL:
        eval_results_root += "-label_CL"
    if args.model_suffix:
        eval_results_root += f"-suffix_{args.model_suffix}"
    
    return eval_results_root

def find_and_summarize_results():
    """查找并总结评估结果"""
    eval_results_path = build_result_path()
    results_file = os.path.join(eval_results_path, "evaluation_results.json")
    
    if not args.verbose:
        pass
    else:
        print(f"Looking for results at: {results_file}")
    
    if not os.path.exists(results_file):
        print(f"[Error] Results file not found: {results_file}")
        return
    
    get_result_function = task2function[args.task_type]
    
    try:
        task_results, avg_score = get_result_function(results_file, task_list[args.task_type])
        
        if args.verbose:
            print(f"\n=== CSR Evaluation Results Summary ===")
            print(f"Model Configuration:")
            print(f"  Base Model: {args.base_model}")
            print(f"  Epochs: {args.epochs}, Batch Size: {args.batch_size}, Learning Rate: {args.lr}")
            print(f"  TopK: {args.topk}, AuxK: {args.auxk}, AuxK Coef: {args.auxk_coef}")
            print(f"  CL Coef: {args.cl_coef}, Use Label CL: {args.use_label_CL}")
            print(f"  Embed Dim: {args.embed_dim}, Hidden Size: {args.hidden_size}")
            if args.initial_topk is not None:
                print(f"  Initial TopK: {args.initial_topk}, K Decay Ratio: {args.k_decay_ratio}")
            print(f"  Model Suffix: {args.model_suffix}")
            print(f"\nTask Type: {args.task_type}")
            print(f"Average Score: {avg_score:.4f}")
            print(f"\nDetailed Results per Task:")
            for task, score in task_results.items():
                print(f"  {task}: {score:.4f}")
            print(f"\nResults file: {results_file}")
        else:
            print(f"\n=== CSR Evaluation Results Summary ===")
            print(f"Average Score: {avg_score:.4f}")
            print(f"Results file: {results_file}")

    except Exception as e:
        print(f"[Error] Failed to process results: {str(e)}")
        if args.verbose:
            import traceback
            print(f"[Error] Traceback: {traceback.format_exc()}")

# 执行主要逻辑
find_and_summarize_results()