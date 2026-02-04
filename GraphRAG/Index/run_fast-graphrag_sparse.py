import asyncio
import os
import logging
import argparse
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from dotenv import load_dotenv
from datasets import load_dataset
from fast_graphrag import GraphRAG
from fast_graphrag._llm import OpenAILLMService
from fast_graphrag._llm._base import BaseEmbeddingService, NoopAsyncContextManager
from tqdm import tqdm
from Evaluation.llm.ollama_client import OllamaClient, OllamaWrapper
from sentence_transformers import SentenceTransformer, SparseEncoder
import numpy as np
import torch
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Load environment variables
load_dotenv()


@dataclass
class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """Embedding service using SentenceTransformer models."""

    embedding_dim: Optional[int] = None
    max_token_size: int = 512
    max_elements_per_request: int = field(default=32)
    model: Any = None
    output_dim: int = 2560

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
        logging.debug("Initialized SentenceTransformerEmbeddingService.")

    async def encode(self, texts: list[str], model: Optional[str] = None) -> np.ndarray:
        try:
            logging.debug(f"Getting embedding for {len(texts)} texts")

            batched_texts = [
                texts[i * self.max_elements_per_request : (i + 1) * self.max_elements_per_request]
                for i in range((len(texts) + self.max_elements_per_request - 1) // self.max_elements_per_request)
            ]
            responses = await asyncio.gather(*[self._embedding_request(batch) for batch in batched_texts])
            embeddings = np.vstack(responses)
            logging.debug(f"Received embedding response: {len(embeddings)} embeddings")
            return embeddings
        except Exception:
            logging.exception("An error occurred during SentenceTransformer embedding.", exc_info=True)
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
                    logging.debug(f"Embedding request for batch size: {len(input_texts)}")
                    
                    # Use torch.no_grad() to prevent gradient accumulation
                    with torch.no_grad():
                        # Use SparseEncoder's encode method
                        embeddings = self.model.encode(
                            input_texts,
                            show_progress_bar=False,
                            batch_size=len(input_texts),
                            convert_to_sparse_tensor=False
                        )
                        
                        # Convert tensor to numpy if needed
                        if isinstance(embeddings, torch.Tensor):
                            embeddings_np = embeddings.cpu().numpy()
                            # Explicitly delete the tensor to free GPU memory
                            del embeddings
                            embeddings = embeddings_np
                        
                        # Slice to use only the first output_dim dimensions
                        if self.output_dim is not None and embeddings.shape[-1] > self.output_dim:
                            embeddings = embeddings[:, :self.output_dim]
                            logging.debug(f"Sliced embeddings to first {self.output_dim} dimensions")
                    
                    # Clear CUDA cache after each batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    return embeddings

# Configuration constants
DOMAIN = "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships."
EXAMPLE_QUERIES = [
    "What is the significance of Christmas Eve in A Christmas Carol?",
    "How does the setting of Victorian London contribute to the story's themes?",
    "Describe the chain of events that leads to Scrooge's transformation.",
    "How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
    "Why does Dickens choose to divide the story into \"staves\" rather than chapters?"
]
ENTITY_TYPES = ["Character", "Animal", "Place", "Object", "Activity", "Event"]

def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    """Group questions by their source"""
    grouped_questions = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions

def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    mode: str,
    model_name: str,
    embedding_model: Any,
    embed_dim: int,
    llm_base_url: str,
    llm_api_key: str,
    questions: Dict[str, List[dict]],
    sample: int
):
    """Process a single corpus: index it and answer its questions"""
    logging.info(f"📚 Processing corpus: {corpus_name}")
    
    # Prepare output directory
    output_dir = f"./results/fast-graphrag/{base_dir}/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")
    
    # Initialize LLM service based on mode
    if mode == "ollama":
        # Create Ollama client
        ollama_client = OllamaClient(base_url=llm_base_url)
        llm_service = OllamaWrapper(ollama_client, model_name)
        logging.info(f"✅ Using Ollama LLM service: {model_name} at {llm_base_url}")
    else:
        # Use OpenAI-compatible service
        llm_service = OpenAILLMService(
            model=model_name,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        logging.info(f"✅ Using OpenAI-compatible LLM service: {model_name} at {llm_base_url}")

    # Initialize GraphRAG
    working_dir = os.path.join(base_dir, corpus_name)
    grag = GraphRAG(
        working_dir=working_dir,
        domain=DOMAIN,
        example_queries="\n".join(EXAMPLE_QUERIES),
        entity_types=ENTITY_TYPES,
        config=GraphRAG.Config(
            llm_service=llm_service,
            embedding_service=SentenceTransformerEmbeddingService(
                model=embedding_model,
                embedding_dim=embed_dim,
                max_token_size=8192,
                output_dim=embed_dim
            ),
        ),
    )
    
    # Index the corpus content
    logging.info(f"📝 Starting to index corpus: {corpus_name}")
    grag.insert(context)
    logging.info(f"✅ Indexed corpus: {corpus_name} ({len(context.split())} words)")
    
    # Ensure data is persisted by waiting for async operations to complete
    import time
    time.sleep(3)  # Give time for data to be written to disk
    logging.info(f"💾 Data persisted to: {working_dir}")
    
    # Verify that data files were created
    if os.path.exists(working_dir):
        files = os.listdir(working_dir)
        logging.info(f"📂 Files in working directory: {files}")
        if not files:
            logging.error(f"❌ No files found in working directory after indexing!")
            return
    else:
        logging.error(f"❌ Working directory does not exist: {working_dir}")
        return
    
    # CRITICAL FIX: Delete the GraphRAG object and recreate it to ensure clean state
    # This forces the object to reload from disk in query mode
    del grag
    logging.info(f"🔄 Recreating GraphRAG object for querying...")
    
    # Reinitialize GraphRAG for query mode - this will load the persisted data
    grag = GraphRAG(
        working_dir=working_dir,
        domain=DOMAIN,
        example_queries="\n".join(EXAMPLE_QUERIES),
        entity_types=ENTITY_TYPES,
        config=GraphRAG.Config(
            llm_service=llm_service,
            embedding_service=SentenceTransformerEmbeddingService(
                model=embedding_model,
                embedding_dim=embed_dim,
                max_token_size=8192,
                output_dim=embed_dim
            ),
        ),
    )
    logging.info(f"✅ GraphRAG object recreated and ready for querying")
    
    # Get questions for this corpus
    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"⚠️ No questions found for corpus: {corpus_name}")
        return
    
    # Sample questions if requested
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]
    
    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")
    
    # Process questions
    results = []
    for q in tqdm(corpus_questions, desc=f"Answering questions for {corpus_name}"):
        try:
            # Execute query
            response = grag.query(q["question"])
            context_chunks = response.to_dict()['context']['chunks']
            contexts = [item[0]["content"] for item in context_chunks]
            predicted_answer = response.response

            # Collect results
            results.append({
                "id": q["id"],
                "question": q["question"],
                "source": corpus_name,
                "context": contexts,
                "evidence": q.get("evidence", ""),
                "question_type": q.get("question_type", ""),
                "generated_answer": predicted_answer,
                "ground_truth": q.get("answer", "")
            })
        except Exception as e:
            logging.error(f"❌ Error processing question {q.get('id')}: {e}")
            results.append({
                "id": q["id"],
                "error": str(e)
            })
    
    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")
    
    # CRITICAL: Clean up to prevent memory leaks
    try:
        # Delete GraphRAG object and its references
        del grag
        del llm_service
        
        # Clear any remaining variables
        del results
        del corpus_questions
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logging.info(f"🧹 Cleaned up memory after processing {corpus_name}")
    except Exception as e:
        logging.warning(f"⚠️ Error during cleanup: {e}")

def main():
    # Define subset paths
    SUBSET_PATHS = {
        "medical": {
            "corpus": "./Datasets/Corpus/medical.parquet",
            "questions": "./Datasets/Questions/medical_questions.parquet"
        },
        "novel": {
            "corpus": "./Datasets/Corpus/novel.parquet",
            "questions": "./Datasets/Questions/novel_questions.parquet"
        }
    }
    
    parser = argparse.ArgumentParser(description="GraphRAG: Process Corpora and Answer Questions")
    
    # Core arguments
    parser.add_argument("--subset", required=True, choices=["medical", "novel"], 
                        help="Subset to process (medical or novel)")
    parser.add_argument("--base_dir", default="./Examples/graphrag_workspace", 
                        help="Base working directory for GraphRAG")
    
    # Model configuration
    parser.add_argument("--mode", choices=["API", "ollama"], default="API",
                        help="Use API or ollama for LLM")
    parser.add_argument("--model_name", default="qwen2.5-14b-instruct", 
                        help="LLM model identifier")
    parser.add_argument("--embed_model_path", default="/home/xzs/data/model/bge-large-en-v1.5", 
                        help="Path to embedding model directory")
    parser.add_argument("--embed_dim", type=int, default=2560,
                        help="Number of dimensions to use from embedding output (default: 2560)")
    parser.add_argument("--sample", type=int, default=None, 
                        help="Number of questions to sample per corpus")
    
    # API configuration
    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1", 
                        help="Base URL for LLM API")
    parser.add_argument("--llm_api_key", default="", 
                        help="API key for LLM service (can also use LLM_API_KEY environment variable)")

    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"graphrag_{args.subset}.log")
        ]
    )
    
    logging.info(f"🚀 Starting GraphRAG processing for subset: {args.subset}")
    
    # Validate subset
    if args.subset not in SUBSET_PATHS:
        logging.error(f"❌ Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
        return
    
    # Get file paths for this subset
    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]
    
    # Handle API key security
    api_key = args.llm_api_key or os.getenv("LLM_API_KEY", "")
    if not api_key:
        logging.warning("⚠️ No API key provided! Requests may fail.")
    
    # Create workspace directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Load corpus data
    try:
        corpus_dataset = load_dataset("parquet", data_files=corpus_path, split="train")
        corpus_data = []
        for item in corpus_dataset:
            corpus_data.append({
                "corpus_name": item["corpus_name"],
                "context": item["context"]
            })
        logging.info(f"📖 Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load corpus: {e}")
        return
    
    # Sample corpus data if requested
    if args.sample:
        corpus_data = corpus_data[:1]
    
    # Initialize embedding model once (shared across all corpora)
    try:
        embedding_model = SparseEncoder(args.embed_model_path)
        logging.info(f"✅ Loaded SentenceTransformer model: {args.embed_model_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load embedding model: {e}")
        return
    
    # Load question data
    try:
        questions_dataset = load_dataset("parquet", data_files=questions_path, split="train")
        question_data = []
        for item in questions_dataset:
            question_data.append({
                "id": item["id"],
                "source": item["source"],
                "question": item["question"],
                "answer": item["answer"],
                "question_type": item["question_type"],
                "evidence": item["evidence"]
            })
        grouped_questions = group_questions_by_source(question_data)
        logging.info(f"❓ Loaded questions with {len(question_data)} entries from {questions_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load questions: {e}")
        return
    
    # Process each corpus sequentially to avoid data race issues
    # GraphRAG uses disk storage that may not handle concurrent access well
    for idx, item in enumerate(corpus_data):
        try:
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing corpus {idx+1}/{len(corpus_data)}: {item['corpus_name']}")
            logging.info(f"{'='*60}\n")
            
            process_corpus(
                item["corpus_name"],
                item["context"],
                args.base_dir,
                args.mode,
                args.model_name,
                embedding_model,  # Pass the shared embedding model
                args.embed_dim,
                args.llm_base_url,
                api_key,
                grouped_questions,
                args.sample,
            )
            
            # Additional cleanup between corpora
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Log memory usage
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logging.info(f"📊 GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                
        except Exception as e:
            logging.exception(f"❌ Failed to process corpus {item['corpus_name']}: {e}")
    
    # Final cleanup
    logging.info("\n🧹 Performing final cleanup...")
    try:
        del embedding_model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("✅ Final cleanup completed")
    except Exception as e:
        logging.warning(f"⚠️ Error during final cleanup: {e}")

if __name__ == "__main__":
    main()