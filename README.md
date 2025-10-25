<div align="center">

<h1 style="font-family: Georgia; font-weight: 600; letter-spacing: 0.5px;">
✨ CSRv2: Unlocking Ultra-Sparse Embeddings ✨
</h1>

<br>

<p style="font-family: Charter, serif; font-size: 15px; line-height: 1.6; color: #444;">
<b>
Lixuan Guo*</b><sup>1,2</sup>,
<a href="https://yifeiwang77.com/" target="_blank"><b>Yifei Wang*</b></a><sup>3</sup>,
<a href="https://neilwen987.github.io/" target="_blank"><b>Tiansheng Wen*</b></a><sup>1,2</sup>,
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

<br>

<img src="./assets/overview.jpg" width="600" style="border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.15);">

<br><br>

<p style="font-size: 15px; color: #444;">
This is the <b>official repository</b> for <b>CSRv2</b>.  
For implementation details and updates, please visit  
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
