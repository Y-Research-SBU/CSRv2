## Instructions for Data Preparation
### Text Embedding
#### Data for CSR training
We use `get_embeddings_of_each_dataset.py` to generate embeddings for each dataset. 

The required parameters are:
- `--model_name`: Backbone for getting embeddings. It needs to be loadable by [sentence_transformers](https://github.com/UKPLab/sentence-transformers).
- `--batch_size`: Batch size for embedding generation. 
- `--save_root_path`: Path to the `npz` file that stores embeddings (and their labels).
- `--dataset_name`: The MTEB dataset to process.
- `--task_type`: Task type of the dataset to process. Note that task type must correspond to task name, such as the task type for banking77 is classification.
- `--gpu`: GPU for embedding generation.

An example for running this code as follows and more commands are available in `dataset_preparation.sh`.
```shell
python get_embeddings_of_each_dataset.py \
    --task_type classification \
    --model_name intfloat/e5-mistral-7b-instruct \
    --batch_size 2 \
    --save_root_path ./original_embeddings/ \
    --dataset_name banking77 \
    --gpu 0
```

We use `get_emmbeddings_for_training.py` to combine tasks of the same task type for task-type-specific evaluation.

The required parameters are:
- `--save_root_path`: Path for saving the combined embeddings.
- `--task_type`: The task type to combine.
- `--embedding_path`: Path for saving embeddings of each dataset.

An example for running this code as follows and more commands are available in `dataset_preparation.sh`.
```shell
python get_embeddings_for_training.py \
    --save_root_path ./embeddings_for_training/Qwen3_4B_without_finetuning \
    --task_type classification \
    --embedding_path ./embeddings_of_each_dataset/Qwen_Qwen3-Embedding-4B
```

#### Data for Backbone Finetuning
We use `get_dataset_for_finetuning.py` to get datasets for finetuning.
For task type except STS, sentence pairs sharing semantically similar information are collected for (MultipleNegativesRankingLoss)[https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss], while for STS, triplets `sentence1-sentence2-score` are collected for (CoSENTLoss)[https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosentloss].These processed datasets can be loaded with Hugging Face (datasets)[https://huggingface.co/docs/datasets/index].

The required parameters are:
- `--task_type`: Task type of the tasks to process.
- `--save_root`: Path to the processed datasets.

Moreover, we can define the number of rows for finetuning `--max_sample_per_dataset`.

An example for running this code as follows:
```shell
python get_dataset_for_finetuning.py \
    --task_type classification \
    --save_root ./MTEB_datasets_for_LLM_finetuning \
    --max_samples_per_dataset 20000
```

We use `combine_datasets_in_mteb_based_on_task_type.py` to combine datasets for task-type backbone finetuning.

The required parameters are:
- `--task_type`: Task type of datasets to combine.
- `--max_rows_per_dataset`: Row number for each task type.
- `--mteb_dataset_path`: Path for storing the final datasets

An example for running this code as follows:
```shell
python combine_datasets_in_mteb_based_on_task_type.py \
    --task_type classification \
    --max_rows_per_dataset 20000 \
    --mteb_dataset_path ./MTEB_datasets_for_LLM_finetuning
```
#### Data for MRL Training
Data for training MRL takes two sets of datasets: [sentence transformers' collection for embedding model training](https://huggingface.co/collections/sentence-transformers/embedding-model-datasets-6644d7a3673a511914aa7552) and MTEB datasets included in the evaluation for fair comparison with CSR training.

We use `combine_datasets_in_sentence_transformers.py` to preprocess sentence transformers data collection. The required parameters are:
- '--max_pairs_per_dataset`: The maximum number of pairs for each dataset in the collection.

An example for runing this code is:
```shell
python combine_datasets_in_sentence_transformers.py \
    --max_pairs_per_dataset 20000 
```

Then we combine datasets for MTEB that have been preprocessed before based on task type with `combine_datasets_in_mteb.py`

```python
python combine_datasets_in_mteb.py
```

Finally, we combine all these with `python combine_large_datasets.py`

```python
python combine_large_datasets.py 
```

### Image Embedding
We generate embeddings for Imagenet1K following the following pipeline:

#### Dataset Download and Preprocess
First, download Imagenet1k dataset and bounding box annotations from [Imagenet1k Official Website](https://www.image-net.org/).

Second, convert dataset to [Pytorch Style](https://github.com/williamFalcon/pytorch-imagenet-dataset) with `annotations.py` and `to_pytorch_style.py`. 

```shell 
python ./dataset_preparation annotations.py --xml_dir "/PATH/TO/TRAIN/ANNOTATION/DIRECTORY" --output_file "/PATH/TO/ANNOTATIONS.TXT"
python ./dataset_preparation to_pytorch_style.py --split_path "/PATH/TO/PYTORCH/STYLE/DATASET"
```

The final dataset should be in the following format:
```
train/
  n01443537/
    images/
      n02058221_0.JPEG
  ...
```

#### Conversion to FFCV Format
Follow the pipeline of [FFCV](https://github.com/libffcv/ffcv) to convert Imagenet1K to FFCV format.
```shell
export IMAGENET_DIR=/path/to/pytorch/format/imagenet/directory/
export WRITE_DIR=/your/path/here/
./write_imagenet.sh "train" 500 0.50 90
./write_imagenet.sh "val" 500 0.50 90
```

#### Embedding Generation with the Selected Backbone
We use `pretrained_embeddings.py` and `stack_emb.py` to generate embeddings with the selected backbone.
```shell
python pretrained_embeddings.py \
	--train_data_ffcv  /PATH/TO/train.ffcv \
	--eval_data_ffcv    /PATH/TO/val.ffcv \
	--model_name "pre-trained visual backbone" \
python stack_emb.py
```