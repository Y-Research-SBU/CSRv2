import os
import re
import shutil
import json
import argparse
from typing import Dict, List

pattern_standard = (
    r'^(?P<base_model>.+?)_(?P<dataset>.+?)_CSR_'
    r'topk_(?P<topk>\d+)_'
    r'auxk_(?P<auxk>\d+)_'
    r'auxk_coef_(?P<auxk_coef>[-.\deE]+)_'
    r'cl_coef_(?P<cl_coef>[-.\deE]+)_'
    r'lr_(?P<lr>[-.\deE]+)_'
    r'hidden_(?P<hidden_size>\d+)'
    r'(?P<label_CL>_label_CL)?'
    r'(?:_(?P<model_suffix>[^_]+))?'  
    r'\.safetensors$'
)

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
    r'(?:_(?P<model_suffix>[^_]+))?'  
    r'(?:_k_decay_(?P<k_decay_ratio>[-.\deE]+))?'
    r'\.safetensors$'
)

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
    r'(?:_(?P<model_suffix>[^_]+))?'  
    r'\.safetensors$'
)

def parse_safetensors_filename(filename: str) -> dict:
    m = re.match(pattern_k_anneal_new, filename)
    if m:
        return m.groupdict()
    
    m = re.match(pattern_k_anneal_old, filename)
    if m:
        return m.groupdict()
    
    m = re.match(pattern_standard, filename)
    if m:
        return m.groupdict()
    
    raise ValueError(f"Filename error: {filename}")


def parse_packaged_model_dirname(dirname: str) -> dict:
    parts = dirname.split('-')
    if len(parts) < 2:
        raise ValueError(f"Invalid packaged model dir: {dirname}")
    
    if len(parts) >= 4:
        base_model = '-'.join(parts[:3])  # Qwen3-Embedding-4B
        dataset = '-'.join(parts[3:])
        return {"base_model": base_model, "dataset": dataset}
    else:
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
        if base_model and parsed.get('base_model') != base_model:
            continue
        
        param_matches = True
        
        has_k_anneal_params_in_search = 'initial_topk' in param_dict or 'k_decay_ratio' in param_dict
        has_k_anneal_params_in_file = parsed.get('initial_topk') is not None or parsed.get('k_decay_ratio') is not None
        
        if has_k_anneal_params_in_search != has_k_anneal_params_in_file:
            param_matches = False
            continue
        
        for k, v in param_dict.items():
            if k == 'use_label_CL':
                if v.lower() == 'true':
                    if not parsed.get('label_CL'):
                        param_matches = False
                        break
                else:
                    if parsed.get('label_CL'):
                        param_matches = False
                        break
            elif k == 'model_suffix':
                parsed_suffix = parsed.get('model_suffix')
                if parsed_suffix != str(v):
                    param_matches = False
                    break
            elif k in ['initial_topk', 'k_decay_ratio']:
                parsed_value = parsed.get(k)
                if parsed_value != str(v):
                    param_matches = False
                    break
            else:
                parsed_value = parsed.get(k)
                if parsed_value is None or parsed_value != str(v):
                    param_matches = False
                    break
        
        if param_matches:
            matches.append(os.path.join(safetensors_dir, fname))
    return matches


def main():
    parser = argparse.ArgumentParser(description="Distribute safetensors models and generate config.json based on parameters")
    parser.add_argument('--packaged_models_root', type=str, required=True, help="Packaged models root path")
    parser.add_argument('--safetensors_dir', type=str, required=True, help="Directory containing safetensors models")
    parser.add_argument('--epochs', type=int, default=10, help="Epoch count (for record keeping only)")
    parser.add_argument('--base_model', type=str, required=True, help="Base model name")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (for record keeping only)")
    parser.add_argument("--lr", type=float, default=4e-5, help="Learning rate (for record keeping only)")
    parser.add_argument('--topk', type=str, required=True, help="topk parameter")
    parser.add_argument('--auxk', type=str, required=True, help="auxk parameter")
    parser.add_argument('--auxk_coef', type=float, default=0.1, help="auxk_coef value (for record keeping only)")
    parser.add_argument('--cl_coef', type=float, default=0.1, help="cl_coef value (for record keeping only)")
    parser.add_argument('--embed_dim', type=int, default=2560, help="Input embedding dimension (for record keeping only)")
    parser.add_argument('--hidden_size', type=int, default=10240, help="Hidden dimension (for record keeping only)")
    parser.add_argument('--model_suffix', type=str, default=None, help="Optional model suffix (for record keeping only)")
    parser.add_argument('--dead_threshold', type=int, default=30, help="dead_threshold parameter")
    parser.add_argument('--input_dim', type=int, default=2560, help="config.json: input_dim")
    parser.add_argument('--hidden_dim', type=int, default=20480, help="config.json: hidden_dim")
    parser.add_argument('--normalize', action='store_true', help="config.json: normalize (default False; include flag to enable)")
    parser.add_argument('--dead_threshold', type=int, default=30, help="config.json: dead_threshold")
    parser.add_argument('--other_param', action='append', default=[], help="Extra parameters in the form param=value; can be supplied multiple times")
    args = parser.parse_args()

    # Parse extra parameters
    other_params = {}
    for p in args.other_param:
        if '=' not in p:
            print(f"Invalid parameter (expected key=value): {p}")
            continue
        k, v = p.split('=', 1)
        other_params[k] = v

    # Compose search parameters
    search_params = other_params.copy()
    search_params['base_model'] = args.base_model
    search_params['topk'] = args.topk
    search_params['auxk'] = args.auxk
    search_params['hidden_size'] = str(args.hidden_dim)

    # Task set to process
    target_tasks = set(args.tasks) if args.tasks else None

    # Iterate through packaged models
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
        
        # Skip tasks not requested by the user
        if target_tasks and dataset_name not in target_tasks:
            print(f"[skip] Task {dataset_name} is not in the requested task list")
            continue
        
        # Build search parameters for the current directory, including model suffix
        current_search_params = search_params.copy()
        if model_suffix:
            current_search_params['model_suffix'] = model_suffix
            
        try:
            matches = find_matching_safetensors(args.safetensors_dir, dataset_name, current_search_params, args.base_model)
        except Exception as e:
            print(f"Error searching safetensors for {dname}: {e}")
            continue
        if not matches:
            print(f"[warn] No matching safetensors model found: dataset={dataset_name}, params={current_search_params}")
            continue
        for safepath in matches:
            target_dir = os.path.join(dpath, "3_SparseAutoEncoder")
            os.makedirs(target_dir, exist_ok=True)
            target_model_path = os.path.join(target_dir, "model.safetensors")
            try:
                shutil.copy2(safepath, target_model_path)
                print(f"[copy] {safepath} -> {target_model_path}")
            except Exception as e:
                print(f"[error] Copy failed: {e}")
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
                print(f"[write] config.json at {config_path}")
                processed_count += 1
            except Exception as e:
                print(f"[error] Failed to write config.json: {e}")
    
    if target_tasks:
        print(f"[done] Processed {processed_count} models for the requested tasks")
        if processed_count < len(target_tasks):
            # Collect processed tasks
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
                print(f"[warn] The following tasks were not processed successfully: {missing_tasks}")

    else:
        print(f"[done] Processed {processed_count} models")

if __name__ == "__main__":
    main()