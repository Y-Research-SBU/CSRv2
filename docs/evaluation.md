## Instructions for Evaluation
### Text Embedding
We provide two ways for text embedding evaluation. One approach is to use the standard library of the [MTEB benchmark](https://github.com/embeddings-benchmark/mteb); the other approach requires first generating the backbone's embedding of the tasks to evaluate, which is suitable for scenarios where multiple evaluations are needed.

#### Evaluation based on `mteb` library
When using the standard library for evaluation, we package the trained model with backbone using [Sparse Encoder](https://sbert.net/docs/package_reference/sparse_encoder/SparseEncoder.html#id1) module in Sentence Transformer. This process is integrated into `all_step_pipeline_mteb_evaluation.py`, whose required parameters are:
- `--eval_tasks`: A list of tasks for evaluation.
- `--packaged_model_dir`: Path to the backbone for packaging.

One example for evaluation based on MTEB library (and training) is as follows, whose training parameters can refer to document for training.

```shell
python all_step_pipeline_mteb_evaluation.py \
    --epochs 10 \
    --eval_tasks arguana cqadupstack-gaming cqadupstack-unix ClimateFEVER_test_top_250_only_w_correct-v2 \
                fiqa nfcorpus scidocs scifact \
    --base_model e5-mistral-7b-instruct \
    --gpu 0 \
    --embed_dim 4096 \
    --hidden_size 16384 \
    --topk 32 \
    --auxk 1024 \
    --auxk_coef 0.1 \
    --lr 0.0001 \
    --model_suffix retrieval \
    --training_embedding_path ./embeddings_for_training/intfloat_without_finetuning/retrieval/train.npz \
    --packaged_model_dir ./e5-mistral-7b-instruct \
    --use_label_CL \
    --initial_topk 64 
```

#### Evaluation based on personalized evaluation functions
One drawback for MTEB-based evaluation is that we must go through the same backbone inference procedure every time we evaluate. If we are evaluating on the same backbone, we can first generate and store backbone's embeddings, which makes evaluation much faster. This process is integrated into `all_step_pipeline_personalized_evaluation.py`, whose required parameters are:
- `--eval_embedding_path`: Path to the pre-generated embeddings.
- `--eval_tasks`: The list of tasks for evaluation.

One example for evaluation based on MTEB library (and training) is as follows, whose training parameters can refer to document for training.

```shell
python all_step_pipeline_personalized_evaluation.py \
    --epochs 10 \
    --eval_tasks arguana cqadupstack-gaming cqadupstack-unix ClimateFEVER_test_top_250_only_w_correct-v2 \
                fiqa nfcorpus scidocs scifact \
    --base_model e5-mistral-7b-instruct \
    --gpu 5 \
    --embed_dim 4096 \
    --hidden_size 16384 \
    --topk 32 \
    --auxk 1024 \
    --auxk_coef 0.1 \
    --lr 0.0001 \
    --eval_embedding_path ./embeddings_of_each_dataset/intfloat_e5-mistral-7b-instruct/ \
    --model_suffix retrieval \
    --training_embedding_path ./embeddings_for_training/intfloat_without_finetuning/retrieval/train.npz
```