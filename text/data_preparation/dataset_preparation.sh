MODEL_NAME=intfloat/e5-mistral-7b-instruct
TASK_TYPE=classification
for DATASET in amazon_massive_intent amazon_massive_scenario amazon_counterfactual mtop_intent mtop_domain imdb \
        tweet_sentiment_extraction emotion toxic_conversations_50k banking77; do
    python get_embeddings_of_each_dataset.py \
    --task_type $TASK_TYPE \
    --model_name $MODEL_NAME \
    --batch_size 2 \
    --save_root_path ./embeddings_of_each_dataset/ \
    --dataset_name $DATASET \
    --gpu 0
done
TASK_TYPE=clustering
for DATASET in arxiv-clustering-s2s biorxiv-clustering-p2p biorxiv-clustering-s2s twentynewsgroups-clustering \
        medrxiv-clustering-p2p medrxiv-clustering-s2s stackexchange-clustering stackexchange-clustering-p2p; do
    python get_embeddings_of_each_dataset.py \
    --task_type $TASK_TYPE \
    --model_name $MODEL_NAME \
    --batch_size 2 \
    --save_root_path ./embeddings_of_each_dataset/ \
    --dataset_name $DATASET \
    --gpu 0
done
TASK_TYPE=retrieval
for DATASET in arguana ClimateFEVER_test_top_250_only_w_correct-v2 cqadupstack-gaming cqadupstack-unix \
        fiqa nfcorpus scidocs scifact; do
    python get_embeddings_of_each_dataset.py \
    --task_type $TASK_TYPE \
    --model_name $MODEL_NAME \
    --batch_size 2 \
    --save_root_path ./embeddings_of_each_dataset/ \
    --dataset_name $DATASET \
    --gpu 0
done
TASK_TYPE=sts
for DATASET in sickr-sts sts12-sts sts13-sts sts14-sts sts15-sts sts16-sts \
        stsbenchmark-sts biosses-sts sts17-crosslingual-sts sts22-crosslingual-sts; do
    python get_embeddings_of_each_dataset.py \
    --task_type $TASK_TYPE \
    --model_name $MODEL_NAME \
    --batch_size 2 \
    --save_root_path ./embeddings_of_each_dataset/ \
    --dataset_name $DATASET \
    --gpu 0
done
TASK_TYPE=reranking
for DATASET in stackoverflowdupquestions-reranking askubuntudupquestions-reranking scidocs-reranking; do
    python get_embeddings_of_each_dataset.py \
    --task_type $TASK_TYPE \
    --model_name $MODEL_NAME \
    --batch_size 2 \
    --save_root_path ./embeddings_of_each_dataset/ \
    --dataset_name $DATASET \
    --gpu 0
done
TASK_TYPE=pair_classification
for DATASET in twitterurlcorpus-pairclassification sprintduplicatequestions-pairclassification; do
    python get_embeddings_of_each_dataset.py \
    --task_type $TASK_TYPE \
    --model_name $MODEL_NAME \
    --batch_size 2 \
    --save_root_path ./embeddings_of_each_dataset/ \
    --dataset_name $DATASET \
    --gpu 0


MODEL_NAME=Qwen/Qwen3-Embedding-4B
TASK_TYPE=reranking
for DATASET in askubuntudupquestions-reranking; do
    python get_embeddings_of_each_dataset.py \
    --task_type $TASK_TYPE \
    --model_name $MODEL_NAME \
    --batch_size 2 \
    --save_root_path ./embeddings_of_each_dataset/ \
    --dataset_name $DATASET \
    --gpu 0
done

for TASK_TYPE in classification clustering retrieval sts reranking pair_classification; do
    python get_embeddings_for_training.py \
        --save_root_path ./embeddings_for_training/intfloat_without_finetuning \
        --task_type $TASK_TYPE \
        --embedding_path ./embeddings_of_each_dataset/intfloat_e5-mistral-7b-instruct
done

for TASK_TYPE in classification clustering retrieval sts reranking pair_classification; do
    python get_embeddings_for_training.py \
        --save_root_path ./embeddings_for_training/Qwen3_4B_without_finetuning \
        --task_type $TASK_TYPE \
        --embedding_path ./embeddings_of_each_dataset/Qwen_Qwen3-Embedding-4B
done