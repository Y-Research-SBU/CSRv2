<div align="center">

<h1 style="font-family: Georgia; font-weight: 600; letter-spacing: 0.5px;">
✨ CSRv2: Unlocking Ultra-Sparse Embeddings ✨
</h1>

<br>

<p style="font-family: Charter, serif; font-size: 15px; line-height: 1.6; color: #444;">
<b>
<a href="https://veritas2024.github.io/" target="_blank"> Lixuan Guo</b><sup>*1,2</sup>,
<a href="https://yifeiwang77.com/" target="_blank"><b>Yifei Wang</b></a><sup>*3</sup>,
<a href="https://neilwen987.github.io/" target="_blank"><b>Tiansheng Wen</b></a><sup>*1,2</sup>,
<a href="https://yfwang.me/" target="_blank"><b>Yifan Wang</b></a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=hFhhrmgAAAAJ&hl=en"><b>Aosong Feng</b></a><sup>4</sup>,</br>
<a href="https://web.xidian.edu.cn/bchen/en/index.html" target="_blank"><b>Bo Chen</b></a><sup>2</sup>,
<a href="https://people.csail.mit.edu/stefje/" target="_blank"><b>Stefanie Jegelka</b></a><sup>3</sup>,
<a href="https://chenyuyou.me/" target="_blank"><b>Chenyu You</b></a><sup>1</sup>
</p>

<p style="font-size: 14px; color: #555; margin-top: 8px;">
<sup>1</sup>Stony Brook University &emsp; 
<sup>2</sup>Xidian University &emsp;
<sup>3</sup>MIT &emsp;
<sup>4</sup>Yale University
</p>

<img src="./assets/overview.jpg" width="600" style="border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.15);">

<br>
<br>

<p style="font-size: 15px; color: #444;">
This is the <b>official repository</b> for <b>CSRv2</b>. For implementation details and updates, please also visit  
<a href="https://github.com/Veritas2024/CSRv2" target="_blank">Lixuan Guo’s GitHub Repository</a>.
</p>

</div>

## &#x1F680; &#x1F680; News
- 2025.10 Code released! Let's explore ultra sparsity together!

In this repo, we will release (**updating**):

- Environment Dependencies 
- Experiment Codes
    - Text Exp on [e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct) and [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B).
        - Inference
        - Data preparation & training. 
    - Image Exp on Imagenet-1K.
        - Inference
        - Data preparation & training.
    - Splade Exp.
    - GraphRAG Exp.
- Checkpoints
    - Text Exp 
    - Image Exp

## Set up
An empty conda environment with Python >= 3.11 is required and install packages according to `requirements.txt`.

```shell
conda create --name csr-v2 python=3.11.13
conda activate csr-v2
pip install -r requirements.txt
```

You can also migrate `conda` environment directly with the `environment.yml`.

```shell
conda env create -f environment.yml
```

## Text Embedding
### Inference with Hugging Face 🤗 [Sentence Transformer](https://www.sbert.net/)
In Sentence Transformers [v5.0.0](https://github.com/UKPLab/sentence-transformers/releases/v5.0.0) release, a new module called SparseEncoder is added, which supports the loading of CSR/CSRv2 models. Our checkpoints will be released in [Y-Research-Group](https://huggingface.co/Y-Research-Group), which can easily be loaded with only a few lines of codes and evaluate on your own datasets.

Demo for generating embeddings based on a CSRv2/CSR model:

```python
from sentence_transformers import SparseEncoder
model = SparseEncoder("/MODEL/NAME")
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
similarities = model.similarity(embeddings, embeddings)
print(similarities)
```

Demo for evaluating a pretrained CSRv2/CSR model on MTEB benchmark datasets.
```python
import mteb
from sentence_transformers import SparseEncoder
model = SparseEncoder(
    "/MODEL/NAME",
    tokenizer_kwargs={"padding_side": "left"},
)
model.prompts = {
    "TASK_NAME": "PROMPT",
}
task_list = mteb.get_task("TASK_NAME")
evaluation = mteb.MTEB(tasks=task_list)
evaluation.run(
    model,
    eval_splits=["test"],
    output_folder="EVAL_RESULT_PATH",
    show_progress_bar=True,
    encode_kwargs={"convert_to_sparse_tensor": False, "batch_size": 2}
)
```

### Data preparation
You need to prepare data for CSRv2 training, backbone finetuning and MRL training for e5-mistral-7b-instruct. Detailed instructions are available in [data preparation instructions](/docs/data_preparation.md).

For CSRv2 training data, you need to execute `get_embeddings_of_each_dataset.py` and `get_embeddings_for_training.py`.

```shell
python get_embeddings_of_each_dataset.py \
    --task_type $TASK_TYPE \
    --model_name $NAME_OF_BACKBONE \
    --batch_size 2 \
    --save_root_path /PATH/TO/SAVE/ROOT/PATH \
    --dataset_name $NAME_OF_DATASET \
    --gpu 0

python get_embeddings_for_training.py \
    --save_root_path /PATH/TO/SAVE/ROOT/PATH \
    --task_type $TASK_TYPE \
    --embedding_path /PATH/TO/SINGLE/TASK/EMBEDDING 
```

For finetuning and MRL training, you need to execute
`get_dataset_for_finetuning.py`, `combine_datasets_in_mteb_based_on_task_type.py` and `combine_datasets_in_sentence_transformers.py`.

```shell
python get_dataset_for_finetuning.py \
    --task_type $TASK_TYPE \
    --save_root /PATH/TO/DATASET \
    --max_samples_per_dataset 20000
python combine_datasets_in_mteb_based_on_task_type.py \
    --task_type $TASK_TYPE \
    --max_rows_per_dataset 20000 \
    --mteb_dataset_path /PATH/TO/DATASET
python combine_datasets_in_sentence_transformers.py \
    --max_pairs_per_dataset 20000 
```

### CSRv2/CSR Training & Evaluation
We have built complete training and evaluation pipeline and you can train and get evaluation results with only one command. We offer two pipelines with different ways for evaluation, with one evaluated with [MTEB library](https://github.com/embeddings-benchmark/mteb) and the other takes our self-built evaluation procedure to avoid unnecessary repetitive backbone embedding inference. Detailed instructions are available in [training instructions](/docs/training.md) and [evaluation instructions](/docs/evaluation.md).

```shell
python all_step_pipeline_mteb_evaluation.py \
    --epochs 10 \
    --eval_tasks $TASKS_TO_EVALUATE \
    --base_model e5-mistral-7b-instruct \
    --gpu 0 \
    --embed_dim 4096 \
    --hidden_size 16384 \
    --topk 32 \
    --auxk 1024 \
    --auxk_coef 0.1 \
    --lr 0.0001 \
    --model_suffix $MODEL_SUFFIX \
    --training_embedding_path /PATH/TO/EMBEDDING/FOR/TRAINING \
    --packaged_model_dir /PATH/TO/PACKAGED/BACKBONE \
    --use_label_CL \
    --initial_topk 64 
```

```shell
python all_step_pipeline_personalized_evaluation.py \
    --epochs 10 \
    --eval_tasks $TASKS_TO_EVALUATE \
    --base_model e5-mistral-7b-instruct \
    --gpu 0 \
    --embed_dim 4096 \
    --hidden_size 16384 \
    --topk 32 \
    --auxk 1024 \
    --auxk_coef 0.1 \
    --lr 0.0001 \
    --eval_embedding_path /PATH/TO/EMBEDDING/FOR/EVALUATION \
    --model_suffix $MODEL_SUFFIX \
    --training_embedding_path /PATH/TO/EMBEDDING/FOR/TRAINING
```

### Finetuning
The complete version of CSRv2 requires backbone finetuning, which can be done with `topk_lora_finetuning.py`. Detailed instructions are available in [training instructions](/docs/training.md).

```shell
CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=31233 topk_lora_finetuning.py \
    --dataset /PATH/TO/DATASET \
    --model_name intfloat/e5-mistral-7b-instruct \
    --loss "multiple_negatives_ranking_loss" \
    --dataset_suffix $SUFFIX \
    --gpu 0,1 \
    --batch_size 4 \
    --gradient_accumulation_steps 32 \
    --topk_k_list "16,32,64,128,256,512,1024,2048,2560" \
    --topk_weights "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0" \
    --max_seq_length 512 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lr 1e-5 \
    --topk_mode "magnitude" \
    --apply_topk_to_backbone \
    --load_from_disk \
    --save_steps 100
```

### GraphRAG evaluation
We evaluate further on [GraphRAG-Bench](https://arxiv.org/abs/2506.05690) with [Fast GraphRAG](https://github.com/HKUDS/LightRAG) framework. Detailed instructions can be found in [GraphRAG evaluation instruction](/docs/GraphRAG_evaluation.md).

```shell
python run_fast-graphrag.py \
  --subset medical \
  --base_dir $WORKSPACE_DIR \
  --model_name gpt-4o-mini \
  --embed_model_path $EMBEDDING_MODEL_PATH \
  --llm_base_url $LLM_BASE_URL

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 -m Evaluation.retrieval_eval \
  --mode API \
  --model gpt-4o-mini \
  --base_url $LLM_BASE_URL \
  --embedding_model $EMBEDDING_MODEL_PATH \
  --data_file $PATH_TO_PREDICTIONS \
  --output_file $PATH_TO_RESULT \
  --detailed_output

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29600 -m Evaluation.generation_eval \
  --mode API \
  --model gpt-4o-mini \
  --base_url $LLM_BASE_URL \
  --embedding_model $EMBEDDING_MODEL_PATH \
  --data_file $PATH_TO_PREDICTIONS \
  --output_file $PATH_TO_RESULT \
  --detailed_output
```

## Image Embedding
### Data preparation
You need to download Imagenet1k and follow the pipeline of [FFCV](https://github.com/libffcv/ffcv-imagenet) to generate the dataset. Details are available in [data preparation instructions](/docs/data_preparation.md).

```shell
python ./dataset_preparation annotations.py --xml_dir "/path/to/train/annotation/directory" --output_file "/path/to/annotation.txt/directory"
python ./dataset_preparation to_pytorch_style.py --split_path "/path/to/pytorch/style/dataset"

cd dataset_preparation
export IMAGENET_DIR=/path/to/pytorch/format/imagenet/directory/
export WRITE_DIR=/your/path/here/
./write_imagenet.sh "train" 500 0.50 90
./write_imagenet.sh "val" 500 0.50 90
```

For training and evaluation simplicity, we extract embeddings before training (except backbone finetuning) and stack embeds together.

```shell
python pretrained_embeddings.py \
	--train_data_ffcv  /path/to/train.ffcv \
	--eval_data_ffcv    /path/to/val.ffcv \
	--model_name "pre-trained visual backbone" 

python stack_emb.py
```

### Training
We train CSR/CSRv2 with `main_visual.py` and the detailed training instructions are available in [training instructions](/docs/training.md).
```shell
python main_visual.py \
    --pretrained_emb ./CSR-precompute-embeds/FF2048_RN50_Embeds/1K_train_ff2048.npz \
    --model_name resnet50d.ra4_e3600_r224_in1k \
    --epochs 10 \
    --initial_topk 64 \
    --topk 32 \
    --auxk 1024 \
    --auxk_coef 0.03125 \
    --cl_coef 0.1 \
    --gpu 0 \
    --model_suffix demo \
    --use_label_CL 
```

### Evaluation
Evaluation takes two steps: get embeddings for evaluation and compute evaluation results. Detailed instruction are available in [evaluation instructions](/docs/evaluation.md).
```shell
python chunk_npz_file.py \
	--input_path "Path/to/original/embeddings" \
	--output_path "Path/to/chunk/directory" \
	--chunk_size "Number of samples per chunk"
python csr_inference.py \
    --train_emb_path  /path/to/train_emb \
    --eval_emb_path    /path/to/val_emb \
    --model_name "pre-trained visual backbone" \
    --topk 8\
    --hidden-size 8192 
    --csr_ckpt "CSR ckpt path"

python ./retrieval/faiss_nn.py --topk $TOPK
python ./retrieval/compute_metrics.py --topk $TOPK  
```

## Citing this paper
If you find this work useful, please cite the accompanying paper:

```shell
@inproceedings{guo2026csrv2,
    title={{CSR}v2: Unlocking Ultra-sparse Embeddings},
    author={Guo, Lixuan and Wang, Yifei and Wen, Tiansheng and Wang, Yifan and Feng, Aosong and Chen, Bo and Jegelka, Stefanie and You, Chenyu},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2026}
}
```

## Acknowledgements
This repository was built off of [CSR](https://github.com/neilwen987/CSR_Adaptive_Rep) and [GraphRAG-Benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark). Thanks for their amazing works!