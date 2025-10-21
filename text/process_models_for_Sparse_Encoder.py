import os
import re
import shutil
import json
import argparse
from typing import Dict, List

# 标准模式（不包含k-annealing参数）
pattern_standard = (
    r'^(?P<base_model>.+?)_(?P<dataset>.+?)_CSR_'
    r'topk_(?P<topk>\d+)_'
    r'auxk_(?P<auxk>\d+)_'
    r'auxk_coef_(?P<auxk_coef>[-.\deE]+)_'
    r'cl_coef_(?P<cl_coef>[-.\deE]+)_'
    r'lr_(?P<lr>[-.\deE]+)_'
    r'hidden_(?P<hidden_size>\d+)'
    r'(?P<label_CL>_label_CL)?'
    r'(?:_(?P<model_suffix>[^_]+))?'  # 添加可选的模型后缀
    r'\.safetensors$'
)

# k-annealing模式1：initial_topk在前面（旧格式）
pattern_k_anneal_old = (
    r'^(?P<base_model>.+?)_(?P<dataset>.+?)_CSR_'
    r'initial_topk_(?P<initial_topk>\d+)_'
    r'topk_(?P<topk>\d+)_'
    r'auxk_(?P<auxk>\d+)_'
    r'auxk_coef_(?P<auxk_coef>[-.\deE]+)_'
    r'cl_coef_(?P<cl_coef>[-.\deE]+)_'
    r'lr_(?P<lr>[-.\deE]+)_'
    r'hidden_(?P<hidden_size>\d+)'
    r'(?P<label_CL>_label_CL)?'
    r'(?:_(?P<model_suffix>[^_]+))?'  # 添加可选的模型后缀
    r'(?:_k_decay_(?P<k_decay_ratio>[-.\deE]+))?'
    r'\.safetensors$'
)

# k-annealing模式2：k-annealing参数在hidden_size后面（新格式）
pattern_k_anneal_new = (
    r'^(?P<base_model>.+?)_(?P<dataset>.+?)_CSR_'
    r'topk_(?P<topk>\d+)_'
    r'auxk_(?P<auxk>\d+)_'
    r'auxk_coef_(?P<auxk_coef>[-.\deE]+)_'
    r'cl_coef_(?P<cl_coef>[-.\deE]+)_'
    r'lr_(?P<lr>[-.\deE]+)_'
    r'hidden_(?P<hidden_size>\d+)_'
    r'initial_topk_(?P<initial_topk>\d+)_'
    r'k_decay_(?P<k_decay_ratio>[-.\deE]+)'
    r'(?P<label_CL>_label_CL)?'
    r'(?:_(?P<model_suffix>[^_]+))?'  # 添加可选的模型后缀
    r'\.safetensors$'
)

def parse_safetensors_filename(filename: str) -> dict:
    # 先尝试新的k-annealing模式（k-annealing参数在hidden_size后面）
    m = re.match(pattern_k_anneal_new, filename)
    if m:
        return m.groupdict()
    
    # 再尝试旧的k-annealing模式（initial_topk在前面，向后兼容）
    m = re.match(pattern_k_anneal_old, filename)
    if m:
        return m.groupdict()
    
    # 最后尝试标准模式（不包含k-annealing参数）
    m = re.match(pattern_standard, filename)
    if m:
        return m.groupdict()
    
    raise ValueError(f"命名格式错误: {filename}")


def parse_packaged_model_dirname(dirname: str) -> dict:
    """解析封装模型目录名，返回 dict: {base_model, dataset, model_suffix}"""
    parts = dirname.split('-')
    if len(parts) < 2:
        raise ValueError(f"Invalid packaged model dir: {dirname}")
    
    # 支持两种格式：
    # 1. 旧格式：{base_model}-{dataset} (如：Qwen3-Embedding-4B-mtop_intent)
    # 2. 新格式：{base_model}-{dataset}-{suffix} (如：Qwen3-Embedding-4B-mtop_intent-test)
    if len(parts) >= 4:
        # 新格式：包含模型后缀
        base_model = '-'.join(parts[:3])  # Qwen3-Embedding-4B
        # 最后一个部分是后缀，中间的是数据集
        dataset = '-'.join(parts[3:])
        return {"base_model": base_model, "dataset": dataset}
    else:
        # 旧格式：不包含模型后缀
        base_model = '-'.join(parts[:3])
        dataset = '-'.join(parts[3:])
        return {"base_model": base_model, "dataset": dataset, "model_suffix": None}

def find_matching_safetensors(
    safetensors_dir: str, 
    dataset: str, 
    param_dict: Dict[str, str],
    base_model: str = None
) -> List[str]:
    print(param_dict)
    """查找匹配的数据集名和所有参数的 safetensors 路径列表"""
    matches = []
    for fname in os.listdir(safetensors_dir):
        if not fname.endswith('.safetensors'):
            continue
        try:
            parsed = parse_safetensors_filename(fname)
        except Exception:
            continue
        
        if parsed['dataset'] != dataset:
            continue
        # 检查base_model是否匹配
        if base_model and parsed.get('base_model') != base_model:
            continue
        
        # 处理 label_cl 参数的特殊逻辑
        param_matches = True
        
        # 首先检查是否需要k-annealing参数匹配
        has_k_anneal_params_in_search = 'initial_topk' in param_dict or 'k_decay_ratio' in param_dict
        has_k_anneal_params_in_file = parsed.get('initial_topk') is not None or parsed.get('k_decay_ratio') is not None
        
        # 如果搜索参数中包含k-annealing参数，则文件必须也包含这些参数
        # 如果搜索参数中不包含k-annealing参数，则文件也不应该包含这些参数
        if has_k_anneal_params_in_search != has_k_anneal_params_in_file:
            param_matches = False
            continue
        
        for k, v in param_dict.items():
            if k == 'use_label_CL':
                # 检查是否使用了标签对比学习
                if v.lower() == 'true':
                    # 期望文件名包含 _label_CL
                    if not parsed.get('label_CL'):
                        param_matches = False
                        break
                else:
                    # 期望文件名不包含 _label_CL
                    if parsed.get('label_CL'):
                        param_matches = False
                        break
            elif k == 'model_suffix':
                # 检查模型后缀是否匹配
                parsed_suffix = parsed.get('model_suffix')
                if parsed_suffix != str(v):
                    param_matches = False
                    break
            elif k in ['initial_topk', 'k_decay_ratio']:
                # 对于k-annealing相关参数，需要精确匹配
                parsed_value = parsed.get(k)
                if parsed_value != str(v):
                    param_matches = False
                    break
            else:
                # 其他参数的常规匹配
                parsed_value = parsed.get(k)
                if parsed_value is None or parsed_value != str(v):
                    param_matches = False
                    break
        
        if param_matches:
            matches.append(os.path.join(safetensors_dir, fname))
    return matches


def main():
    parser = argparse.ArgumentParser(description="根据参数分发 safetensors 模型及生成 config.json")
    parser.add_argument('--packaged_models_root', type=str, required=True, help="封装模型父路径")
    parser.add_argument('--safetensors_dir', type=str, required=True, help="safetensors模型文件夹路径")
    parser.add_argument('--epochs', type=int, default=10, help="训练轮数，仅用于记录")
    parser.add_argument('--base_model', type=str, required=True, help="base model 名称")
    parser.add_argument("--batch_size", type=int, default=128, help="训练批次大小，仅用于记录")
    parser.add_argument("--lr", type=float, default=4e-5, help="学习率，仅用于记录")
    parser.add_argument('--topk', type=str, required=True, help="topk 参数")
    parser.add_argument('--auxk', type=str, required=True, help="auxk 参数")
    parser.add_argument('--auxk_coef', type=float, default=0.1, help="auxk_coef 参数，仅用于记录")
    parser.add_argument('--cl_coef', type=float, default=0.1, help="cl_coef 参数，仅用于记录")
    parser.add_argument('--embed_dim', type=int, default=2560, help="输入嵌入维度，仅用于记录")
    parser.add_argument('--hidden_size', type=int, default=10240, help="隐藏层维度，仅用于记录")
    parser.add_argument('--model_suffix', type=str, default=None, help="模型后缀，可选，仅用于记录")
    parser.add_argument('--dead_threshold', type=int, default=30, help="dead_threshold 参数")
    """默认参数"""
    parser.add_argument('--input_dim', type=int, default=2560, help="config.json: input_dim")
    parser.add_argument('--hidden_dim', type=int, default=20480, help="config.json: hidden_dim")
    parser.add_argument('--normalize', action='store_true', help="config.json: normalize (默认为False，带此参数为True)")
    parser.add_argument('--dead_threshold', type=int, default=30, help="config.json: dead_threshold")
    parser.add_argument('--other_param', action='append', default=[], help="其他参数，格式为 param=value，可添加多个")
    args = parser.parse_args()

    # 解析其它参数
    other_params = {}
    for p in args.other_param:
        if '=' not in p:
            print(f"非法参数（应为key=value格式）: {p}")
            continue
        k, v = p.split('=', 1)
        other_params[k] = v

    # 合成查找参数
    search_params = other_params.copy()
    search_params['base_model'] = args.base_model
    search_params['topk'] = args.topk
    search_params['auxk'] = args.auxk
    search_params['hidden_size'] = str(args.hidden_dim)

    # 要处理的任务集合
    target_tasks = set(args.tasks) if args.tasks else None

    # 遍历封装模型
    processed_count = 0
    for dname in os.listdir(args.packaged_models_root):
        dpath = os.path.join(args.packaged_models_root, dname)
        if not os.path.isdir(dpath):
            continue
        try:
            parsed_dir = parse_packaged_model_dirname(dname)
        except Exception:
            continue
        if parsed_dir['base_model'] != args.base_model:
            continue
        
        dataset_name = parsed_dir['dataset']
        model_suffix = parsed_dir.get('model_suffix')
        
        # 如果指定了任务列表，则只处理指定的任务
        if target_tasks and dataset_name not in target_tasks:
            print(f"[跳过] 任务 {dataset_name} 不在指定的任务列表中")
            continue
        
        # 为当前目录构建搜索参数，包含模型后缀
        current_search_params = search_params.copy()
        if model_suffix:
            current_search_params['model_suffix'] = model_suffix
            
        try:
            matches = find_matching_safetensors(args.safetensors_dir, dataset_name, current_search_params, args.base_model)
        except Exception as e:
            print(f"Error searching safetensors for {dname}: {e}")
            continue
        if not matches:
            print(f"[警告] 未找到匹配的safetensors模型: dataset={dataset_name}, params={current_search_params}")
            continue
        for safepath in matches:
            target_dir = os.path.join(dpath, "3_SparseAutoEncoder")
            os.makedirs(target_dir, exist_ok=True)
            target_model_path = os.path.join(target_dir, "model.safetensors")
            try:
                shutil.copy2(safepath, target_model_path)
                print(f"[复制] {safepath} -> {target_model_path}")
            except Exception as e:
                print(f"[错误] 复制失败: {e}")
                continue
            config = {
                "input_dim": args.input_dim,
                "hidden_dim": args.hidden_dim,
                "k": int(args.topk),
                "k_aux": int(args.auxk),
                "normalize": args.normalize,
                "dead_threshold": args.dead_threshold
            }
            config_path = os.path.join(target_dir, "config.json")
            try:
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                print(f"[写入] config.json at {config_path}")
                processed_count += 1
            except Exception as e:
                print(f"[错误] 写入config.json失败: {e}")
    
    if target_tasks:
        print(f"[完成] 已处理 {processed_count} 个指定任务的模型")
        if processed_count < len(target_tasks):
            # 获取已处理的任务列表
            processed_tasks = set()
            for dname in os.listdir(args.packaged_models_root):
                if not os.path.isdir(os.path.join(args.packaged_models_root, dname)):
                    continue
                try:
                    parsed_dir = parse_packaged_model_dirname(dname)
                    if parsed_dir['base_model'] == args.base_model and parsed_dir['dataset'] in target_tasks:
                        config_path = os.path.join(args.packaged_models_root, dname, "3_SparseAutoEncoder", "config.json")
                        if os.path.exists(config_path):
                            processed_tasks.add(parsed_dir['dataset'])
                except Exception:
                    continue
            
            missing_tasks = target_tasks - processed_tasks
            if missing_tasks:
                print(f"[警告] 以下任务未成功处理: {missing_tasks}")

# 测试正则表达式解析（包含模型后缀）
# test_filename = "Qwen3-Embedding-4B_banking77_CSR_topk_32_auxk_1024_auxk_coef_0.1_cl_coef_0.1_lr_4e-05_hidden_10240_test.safetensors"
# print("测试文件名解析:", parse_safetensors_filename(test_filename))
    else:
        print(f"[完成] 已处理 {processed_count} 个模型")

if __name__ == "__main__":
    main()