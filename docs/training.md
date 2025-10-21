## Instructions for Training
### Text Embedding
#### CSR training 
We use `CSR_training.py` for training CSR and its variants.

The required parameters are:
- `--pretrained_emb`: Path to the embedding file for CSR training. (Note that the path must point to a `npz` file.)
- `--embed_dim`: Dimension of the embedding for training.
- `--hidden_size`: Dimension of hidden latent in CSR.
- `--gpu`: The GPU used for training.
- `--base_model`: Name of the backbone for generating embeddings.
- `--model_suffix`: Suffix added to the end of checkpoints for easier search.
- `--epochs`: Number of epochs for training.
- `--lr`: Learning rate for training.
- `--topk`: TopK selection in CSR training.
- `--use_CL`: Add contrastive loss to training (i.e. training CSR instead of SAE).
- `--cl_coef`: Cofficient of Contrastive Loss.
- `--auxk`: Number of auxk in Auxk Loss.
- `--auxk_coef`: Coefficient of Auxk Loss.

To add anneal to CSR, we need to define two parameters:
- `--initial_top`: TopK at the start of training.
- `--k_decay_ratio`: A float between 0 and 1 to define when K converges to `--topk`.

To add supervised contrastive loss to training, we need to add parameter:
- `--use_label_CL`

Note that this file is executed as a subprocess of `all_step_pipeline_mteb_evaluation.py` and `all_step_pipeline_personalized_evaluation.py`. Names of the parameters are the same.

#### MRL Training
We use `train_mrl_with_sentence_transformers.py` to train MRL on embedding models that do not support MRL.

The required parameters:
- `--model_name`: Name of the embedding model to finetune.
- `--dataset`: Path to the dataset for traiing. 
- `--lora_alpha`: $\alpha$ in Lora.
- `--lora_r`: $r$ in Lora.
- `--lr`: Learning rate in training.
- `--batch_size`: Batch size for finetuning.
- `--loss`: Name of loss for finetuning.
- `--gradient_accumulation_steps`: Number of step of gradient accumulation, whose goal is to achieve larger batch size in limited GPU memory.
- `--dataset_suffix`: Suffix added in saving path.


One example is:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 torchrun --nproc_per_node=7 --master_port=31233 train_mrl_with_sentence_transformers.py \
    --model_name intfloat/e5-mistral-7b-instruct \
    --dataset_suffix mrl-0815 \
    --lora_alpha 16 \
    --batch_size 32 \
    --lr 2e-5 \
    --gpu 0,1,2,3,4,5,6 \
    --load_from_disk \
    --loss multiple_negatives_ranking_loss \
    --gradient_accumulation_steps 4 \
    --dataset ./mrl_dataset
```

#### Backbone Finetuning
We use `topk_lora_finetuning.py` to finetune backbone for further performance improvement.

The required parameters are:
- `--dataset`: Path to the dataset backbone is finetuned on.
- `--model_name`: Model to finetune. This model must be loaded with sentence transformers.
- `--loss`: Loss function used in training. `multiple_negative_ranking_loss` is used for all task-type finetuning except STS, whose loss is `cosent`.
- `--gpu`: Used GPU(s).
- `--batch_size`: Batch size.
- `--gradient_accumulation_steps: Number of step of gradient accumulation, whose goal is to achieve larger batch size in limited GPU memory.
- `topk_k_list`: List of TopK values.
- `topk_weights`: Weights of each TopK in `topk_k_list`.
- `max_seq_length`: Maximum sequence length as input.
- `--lora_r` and `--lora_alpha`: Settings of Lora.
- `--lr`: Learning rate for training.
- `--topk_mode`: Way to define "topk", i.e. whether take the absolute value of features before topk masking.
- `--save_steps`: Number of steps between each time the model weights are saved.

One example is:
```shell
CUDA_VISIBLE_DEVICES=0,1
for TASK in classification; do
    torchrun --nproc_per_node=2 --master_port=31233 topk_lora_finetuning.py \
        --dataset ./MTEB_single_task_type_finetuning_dataset/${TASK} \
        --model_name intfloat/e5-mistral-7b-instruct \
        --loss "multiple_negatives_ranking_loss" \
        --dataset_suffix "${TASK}_intfloat" \
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
done
```

### Image embedding
We use `main_visual.py` for CSRv2/CSR training.

The required parameters are:
- `--pretrained_emb`: Path to the embedding file for CSR training. (Note that the path must point to a `npz` file.)
- `--embed_dim`: Dimension of the embedding for training.
- `--hidden_size`: Dimension of hidden latent in CSR.
- `--gpu`: The GPU used for training.
- `--base_model`: Name of the backbone for generating embeddings.
- `--model_suffix`: Suffix added to the end of checkpoints for easier search.
- `--epochs`: Number of epochs for training.
- `--lr`: Learning rate for training.
- `--topk`: TopK selection in CSR training.
- `--use_CL`: Add contrastive loss to training (i.e. training CSR instead of SAE).
- `--cl_coef`: Cofficient of Contrastive Loss.
- `--auxk`: Number of auxk in Auxk Loss.
- `--auxk_coef`: Coefficient of Auxk Loss.

To add anneal to CSR, we need to define two parameters:
- `--initial_top`: TopK at the start of training.
- `--k_decay_ratio`: A float between 0 and 1 to define when K converges to `--topk`.

To add supervised contrastive loss to training, we need to add parameter:
- `--use_label_CL`

One training example is as follows:

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