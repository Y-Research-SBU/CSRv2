import asyncio
import argparse
import json
import numpy as np
import os
import torch
from typing import Dict, List, Tuple, Any, Optional
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from datasets import Dataset
from langchain_openai import ChatOpenAI
from sentence_transformers import SparseEncoder
from Evaluation.metrics import compute_answer_correctness, compute_coverage_score, compute_faithfulness_score, compute_rouge_score
from Evaluation.llm import OllamaClient, OllamaWrapper

SEED = 42


class SparseEncoderEmbeddings(Embeddings):
    """Custom Embeddings wrapper for SparseEncoder that implements Langchain's Embeddings interface"""
    
    def __init__(self, model: SparseEncoder, output_dim: Optional[int] = None):
        self.model = model
        self.output_dim = output_dim
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                batch_size=len(texts),
                convert_to_sparse_tensor=False
            )
            
            # Convert tensor to numpy if needed
            if isinstance(embeddings, torch.Tensor):
                embeddings_np = embeddings.cpu().numpy()
                del embeddings
                embeddings = embeddings_np
            
            # Slice to use only the first output_dim dimensions
            if self.output_dim is not None and embeddings.shape[-1] > self.output_dim:
                embeddings = embeddings[:, :self.output_dim]
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        with torch.no_grad():
            embeddings = self.model.encode(
                [text],
                show_progress_bar=False,
                batch_size=1,
                convert_to_sparse_tensor=False
            )
            
            # Convert tensor to numpy if needed
            if isinstance(embeddings, torch.Tensor):
                embeddings_np = embeddings.cpu().numpy()
                del embeddings
                embeddings = embeddings_np
            
            # Slice to use only the first output_dim dimensions
            if self.output_dim is not None and embeddings.shape[-1] > self.output_dim:
                embeddings = embeddings[:, :self.output_dim]
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return embeddings[0].tolist()


async def evaluate_dataset(
    dataset: Dataset,
    metrics: List[str],
    llm: Any,
    embeddings: Embeddings,
    max_concurrent: int = 3,  # Limit concurrent evaluations
    detailed_output: bool = False
) -> Dict[str, Any]:
    """Evaluate the metric scores on the entire dataset."""
    results = {metric: [] for metric in metrics}
    detailed_results = [] if detailed_output else None
    
    ids = dataset["id"]
    questions = dataset["question"]
    answers = dataset["answer"]
    contexts_list = dataset["contexts"]
    ground_truths = dataset["ground_truth"]
    
    total_samples = len(questions)
    print(f"\nStarting evaluation of {total_samples} samples...")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_semaphore(i):
        async with semaphore:
            sample_metrics = await evaluate_sample(
                question=questions[i],
                answer=answers[i],
                contexts=contexts_list[i],
                ground_truth=ground_truths[i],
                metrics=metrics,
                llm=llm,
                embeddings=embeddings
            )
            if detailed_output:
                return {
                    "id": ids[i],
                    "question": questions[i],
                    "ground_truth": ground_truths[i],
                    "generated_answer": answers[i],
                    "contexts": contexts_list[i],
                    "metrics": sample_metrics
                }
            return sample_metrics

    tasks = [evaluate_with_semaphore(i) for i in range(total_samples)]
    
    sample_results = []
    completed = 0

    for future in asyncio.as_completed(tasks):
        try:
            result = await future
            if detailed_output and detailed_results is not None:
                detailed_results.append(result)
                # metrics aggregation (guard types for linters)
                if isinstance(result, dict):
                    metrics_dict = result.get("metrics")
                    if isinstance(metrics_dict, dict):
                        for metric, score in metrics_dict.items():
                            if isinstance(score, (int, float)) and not np.isnan(score):
                                results[metric].append(score)
            else:
                sample_results.append(result)
                if isinstance(result, dict):
                    for metric, score in result.items():
                        if isinstance(score, (int, float)) and not np.isnan(score):
                            results[metric].append(score)
            completed += 1
            print(f"✅ Completed sample {completed}/{total_samples} - {(completed/total_samples)*100:.1f}%")
        except Exception as e:
            print(f"❌ Sample failed: {e}")
            completed += 1
    
    avg_results = {metric: np.nanmean(scores) if scores else 0.0 for metric, scores in results.items()}
    
    if detailed_output:
        return {
            "average_scores": avg_results,
            "detailed": detailed_results
        }
    else:
        return avg_results


async def evaluate_sample(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: str,
    metrics: List[str],
    llm: Any,
    embeddings: Embeddings
) -> Dict[str, float]:
    """Evaluate the metric scores for a single sample."""
    results = {}
    
    tasks = {}
    if "rouge_score" in metrics:
        tasks["rouge_score"] = compute_rouge_score(answer, ground_truth)
    
    if "answer_correctness" in metrics:
        tasks["answer_correctness"] = compute_answer_correctness(
            question, answer, ground_truth, llm, embeddings
        )
    
    if "coverage_score" in metrics:
        tasks["coverage_score"] = compute_coverage_score(
            question, ground_truth, answer, llm
        )
    
    if "faithfulness" in metrics:
        tasks["faithfulness"] = compute_faithfulness_score(
            question, answer, contexts, llm
        )
    
    task_results = await asyncio.gather(*tasks.values())
    
    for i, metric in enumerate(tasks.keys()):
        results[metric] = task_results[i]
    
    return results


async def main(args: argparse.Namespace):
    """Main evaluation function that accepts command-line arguments."""
    if args.mode == "API":
        # Check if the API key is set
        if not os.getenv("LLM_API_KEY"):
            raise ValueError("LLM_API_KEY environment variable is not set")
    
        # Initialize the model
        # Wrap API key in SecretStr to satisfy type hints
        from pydantic import SecretStr
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY environment variable is not set")
        llm = ChatOpenAI(
            model=args.model,
            base_url=args.base_url,
            api_key=SecretStr(api_key),
            temperature=0.0,
            max_retries=3,
            timeout=30,
            model_kwargs={
                "top_p": 1,
                "seed": SEED,
                "presence_penalty": 0,
                "frequency_penalty": 0
            }
        )
        
        # Initialize the SparseEncoder embedding model
        print(f"Loading SparseEncoder model from {args.embedding_model}...")
        sparse_encoder = SparseEncoder(args.embedding_model)
        embedding = SparseEncoderEmbeddings(
            model=sparse_encoder,
            output_dim=args.embed_dim
        )
        print(f"✅ Loaded SparseEncoder model with output_dim={args.embed_dim}")
    
    elif args.mode == "ollama":
        ollama_client = OllamaClient(base_url=args.base_url)
        llm = OllamaWrapper(
            ollama_client,
            args.model,
            default_options={
                "temperature": 0.0,
                "top_p": 1,
                "num_ctx": 32768,
                "seed": SEED
            }
        )
        
        # Initialize the SparseEncoder embedding model for ollama mode
        print(f"Loading SparseEncoder model from {args.embedding_model}...")
        sparse_encoder = SparseEncoder(args.embedding_model)
        embedding = SparseEncoderEmbeddings(
            model=sparse_encoder,
            output_dim=args.embed_dim
        )
        print(f"✅ Loaded SparseEncoder model with output_dim={args.embed_dim}")
    
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    # Load evaluation data
    print(f"Loading evaluation data from {args.data_file}...")
    with open(args.data_file, 'r') as f:
        file_data = json.load(f)  # Now a list of question items
    
    # Define the evaluation metrics for each question type
    metric_config = {
        'Fact Retrieval': ["rouge_score", "answer_correctness"],
        'Complex Reasoning': ["rouge_score", "answer_correctness"],
        'Contextual Summarize': ["answer_correctness", "coverage_score"],
        'Creative Generation': ["answer_correctness", "coverage_score", "faithfulness"]
    }
    
    # Group data by question type
    grouped_data = {}
    for item in file_data:
        q_type = item.get("question_type", "Uncategorized")
        if q_type not in grouped_data:
            grouped_data[q_type] = []
        grouped_data[q_type].append(item)
    
    all_results = {}
    
    # Evaluate each found question type (only those in metric_config)
    for question_type in list(grouped_data.keys()):
        # Skip types not defined in metric_config
        if question_type not in metric_config:
            print(f"Skipping undefined question type: {question_type}")
            continue
            
        print(f"\n{'='*50}")
        print(f"Evaluating question type: {question_type}")
        print(f"{'='*50}")
        
        # Prepare data from grouped items
        group_items = grouped_data[question_type]
        ids = [item['id'] for item in group_items]
        questions = [item['question'] for item in group_items]
        ground_truths = [item['ground_truth'] for item in group_items]
        answers = [item['generated_answer'] for item in group_items]
        contexts = [item['context'] for item in group_items]
        
        # Create dataset
        data = {
            "id": ids,
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(data)

        # If sample
        if args.num_samples:
            dataset = dataset.select([i for i in list(range(min(args.num_samples, len(dataset))))])

        # Perform evaluation
        results = await evaluate_dataset(
            dataset=dataset,
            metrics=metric_config[question_type],
            llm=llm, 
            embeddings=embedding,
            detailed_output=args.detailed_output
        )
        
        all_results[question_type] = results
        print(f"\nResults for {question_type}:")
        if args.detailed_output:
            for metric, score in results["average_scores"].items():
                print(f"  {metric}: {score:.4f}")
        else:
            for metric, score in results.items():
                print(f"  {metric}: {score:.4f}")
    
    # Save final results
    if args.output_file:
        print(f"\nSaving results to {args.output_file}...")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    await llm.close() if args.mode == "ollama" else None
    print('\nEvaluation complete.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RAG performance using various metrics with SparseEncoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode", 
        required=True,
        choices=["API", "ollama"],
        type=str,
        default="API",
        help="Use API or ollama for LLM"
    )

    parser.add_argument(
        "--model", 
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for evaluation"
    )
    
    parser.add_argument(
        "--base_url", 
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for the OpenAI API"
    )
    
    parser.add_argument(
        "--embedding_model", 
        type=str,
        required=True,
        help="Path to SparseEncoder model directory"
    )
    
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=2560,
        help="Number of dimensions to use from embedding output (truncate to first N dimensions)"
    )
    
    parser.add_argument(
        "--data_file", 
        type=str,
        required=True,
        help="Path to JSON file containing evaluation data"
    )
    
    parser.add_argument(
        "--output_file", 
        type=str,
        default="generation_results_sparse.json",
        help="Path to save evaluation results"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to use for evaluation"
    )

    parser.add_argument(
        "--detailed_output",
        action="store_true",
        help="Whether to include detailed output"
    )
    
    args = parser.parse_args()
    
    asyncio.run(main(args))
