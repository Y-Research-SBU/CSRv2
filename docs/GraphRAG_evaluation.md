## Instructions for GraphRAG Evaluation
### Environtment Preparation
Since fast-graphrag does not support HuggingFace Embedding, we first need to adjust code according to [GraphRAG-Bench](https://arxiv.org/abs/2506.05690)'s instruction. The detailed process is as follows:

Add support for HuggingFace Embedding by creating a new file named `_hf.py` under `fast_graphrag/_llm` directory.

```python
import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from fast_graphrag._utils import logger
from fast_graphrag._llm._base import BaseEmbeddingService, NoopAsyncContextManager

@dataclass
class HuggingFaceEmbeddingService(BaseEmbeddingService):
    """Embedding service using HuggingFace models."""

    embedding_dim: Optional[int] = None  # Can be set dynamically if needed
    max_token_size: int = 512
    max_elements_per_request: int = field(default=32)
    tokenizer: Any = None
    model: Any = None

    def __post_init__(self):
        self.embedding_max_requests_concurrent = (
            asyncio.Semaphore(self.max_requests_concurrent) if self.rate_limit_concurrency else NoopAsyncContextManager()
        )
        self.embedding_per_minute_limiter = (
            AsyncLimiter(self.max_requests_per_minute, 60) if self.rate_limit_per_minute else NoopAsyncContextManager()
        )
        self.embedding_per_second_limiter = (
            AsyncLimiter(self.max_requests_per_second, 1) if self.rate_limit_per_second else NoopAsyncContextManager()
        )
        logger.debug("Initialized HuggingFaceEmbeddingService.")

    async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray:
        try:
            logger.debug(f"Getting embedding for texts: {texts}")

            batched_texts = [
                texts[i * self.max_elements_per_request : (i + 1) * self.max_elements_per_request]
                for i in range((len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request)
            ]
            responses = await asyncio.gather(*[self._embedding_request(batch) for batch in batched_texts])
            embeddings = np.vstack(responses)
            logger.debug(f"Received embedding response: {len(embeddings)} embeddings")
            return embeddings
        except Exception:
            logger.exception("An error occurred during HuggingFace embedding.", exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RuntimeError, torch.cuda.CudaError)),
    )
    async def _embedding_request(self, input_texts: list[str]) -> np.ndarray:
        async with self.embedding_max_requests_concurrent:
            async with self.embedding_per_minute_limiter:
                async with self.embedding_per_second_limiter:
                    logger.debug(f"Embedding request for batch size: {len(input_texts)}")
                    device = (
                        next(self.model.parameters()).device if torch.cuda.is_available()
                        else torch.device("mps") if torch.backends.mps.is_available()
                        else torch.device("cpu")
                    )
                    self.model = self.model.to(device)

                    encoded = self.tokenizer(
                        input_texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_token_size
                    ).to(device)

                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=encoded["input_ids"],
                            attention_mask=encoded["attention_mask"]
                        )
                        embeddings = outputs.last_hidden_state.mean(dim=1)

                    if embeddings.dtype == torch.bfloat16:
                        return embeddings.detach().to(torch.float32).cpu().numpy()
                    else:
                        return embeddings.detach().cpu().numpy()

```

Include HuggingFaceEmbedding initialization by modifying `fast_graphrag/_llm/__init__.py`
```python
__all__ = [
    ...
    "HuggingFaceEmbeddingService",
]
...
from ._hf import HuggingFaceEmbeddingService
```

### Index
Backbone (e.g. Qwen3-Embedding-4B) and CSR models are indexed seperately, as backbone need to be loaded with `Sentence_Transformer` while CSR models need to be loaded with `SparseEncoder`.

Backbone indexing is executed with `run_fast-graphrag.py`. The required parameters are:
- `--subset`: The subset you want to index on. Avaiable subsets are `medical` and `novel`.
- `--base_dir`: Work directory for Fast-GraphRAG.
- `--embed_model_path`: Backbone name for indexing, which must be suported by `Sentence transformer`.
- `--model_name`: Model name for the API-loaded LLM (e.g. GPT-4o-mini)
- `--embed_dim`: Truncated dim of the embedding generated by `embed_model_path`. If you want full backbone inference, set it the same as backbone output dim. If you want MRL inference, set it the truncated dim you would like to set for MRL.
- `--llm_base_url`: Base url for LLM API. We set default `https://api.openai.com/v1` while you can also choose other API platform if you like.

One example for Qwen3-Embedding-4B (`embed_dim=2560`) with MRL `truncated_dim=32` on novel subset is:
```shell
export LLM_API_KEY=$YOUR_API_KEY
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 ./Index/run_fast-graphrag.py \
  --subset novel \
  --base_dir ./Qwen_MRL_32 \
  --model_name gpt-4o-mini \
  --embed_model_path Qwen/Qwen3-Embedding-4B \
  --llm_base_url https://api.openai.com/v1 \
  --embed_dim 32
```

CSR index is similar except that `--embed_dim` must be set the same with CSR's `hidden_dim`. (sparsity level `K` needs to be set in CSR's `config.json`)

One example for CSR with Qwen3-Embedding-4B as backbone and `hidden_dim=10240` on novel subset is:
```shell
export LLM_API_KEY=$YOUR_API_KEY
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 ./Index/run_fast-graphrag_sparse.py \
  --subset novel \
  --base_dir ./Qwen_CSR \
  --model_name gpt-4o-mini \
  --embed_model_path $PATH_TO_CSR_MODEL \
  --llm_base_url https://api.openai.com/v1 \
  --embed_dim 10240
```

### Evaluation
The pipelines for generation evaluation and retrieval evaluation are very similar. For generation evaluation of full/MRL-truncated backbone, we execute `generation_eval.py` with following parameters to be set:
- `--model`: The API model used for evaluation (e.g. `gpt-4o-mini`)
- `--base_url`: Base url for API platform
- `--embedding_model`: Path to embedding model that can be loaded with `SentenceTransformer`
- `--embed_dim`: Dim for the generated embeddings. Set backbone's output dim when conducting full backbone evaluation or `truncated_dim` when conducting MRL evaluation.
- `--data_file`: Path to JSON file containing evaluation data (as produced by index procedure)
- `--output_file`: Path to save evaluation results

One example for Qwen3-Embedding-4B (`embed_dim=2560`) with MRL `truncated_dim=32` on novel subset is:
```shell
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 -m Evaluation.generation_eval \
  --mode API \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --embedding_model Qwen/Qwen3-Embedding-4B \
  --data_file $PATH_TO_NOVEL_PREDICTIONS \
  --output_file $PATH_TO_EVALUATION_RESULTS \
  --detailed_output \
  --embed_dim 32
```

CSR generation evaluation is similar except that `--embed_dim` must be set the same with CSR's `hidden_dim`. (sparsity level `K` needs to be set in CSR's `config.json`)

One example for CSR with Qwen3-Embedding-4B as backbone and `hidden_dim=10240` on novel subset is:
```shell
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 -m Evaluation.generation_eval_sparse \
  --mode API \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --embedding_model $PATH_TO_CSR_MODEL \
  --data_file $PATH_TO_NOVEL_PREDICTIONS \
  --output_file $PATH_TO_EVALUATION_RESULTS \
  --detailed_output \
  --embed_dim 10240
```

Retrieval evaluation follows the same pipeline except that you need to execute files `retrieval_eval.py` and `retrieval_eval_sparse.py`. Demo scripts are as follows:
```shell
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 -m Evaluation.retrieval_eval \
  --mode API \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --embedding_model Qwen/Qwen3-Embedding-4B \
  --data_file $PATH_TO_NOVEL_PREDICTIONS \
  --output_file $PATH_TO_EVALUATION_RESULTS \
  --detailed_output \
  --embed_dim 32
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29500 -m Evaluation.retrieval_eval_sparse \
  --mode API \
  --model gpt-4o-mini \
  --base_url https://api.openai.com/v1 \
  --embedding_model $PATH_TO_CSR_MODEL \
  --data_file $PATH_TO_NOVEL_PREDICTIONS \
  --output_file $PATH_TO_EVALUATION_RESULTS \
  --detailed_output \
  --embed_dim 10240
```