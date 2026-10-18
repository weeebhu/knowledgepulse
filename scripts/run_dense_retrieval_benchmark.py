import argparse
import json
from pathlib import Path

from knowledgepulse.benchmarks.dataset import load_benchmark_queries
from knowledgepulse.embeddings.sentence_transformer import SentenceTransformerProvider
from knowledgepulse.evaluation.retrieval_benchmark import run_retrieval_benchmark
from knowledgepulse.ingestion.pdf import ingest_pdf
from knowledgepulse.pipeline.knowledge_pipeline import KnowledgePipeline

GOLDEN_DIR = Path("datasets/golden")
DATASET_PATH = GOLDEN_DIR / "retrieval" / "queries.json"
RESULTS_DIR = Path("experiments/retrieval")
RESULTS_PATH = RESULTS_DIR / "dense_baseline.json"

DOCUMENT_PATHS = [
    GOLDEN_DIR / "DOC-001_Production_Deployment_Runbook.pdf",
    GOLDEN_DIR / "DOC-002_CICD_Pipeline_Migration_Guide.pdf",
    GOLDEN_DIR / "DOC-003_Database_Backup_and_Retention_Policy.pdf",
    GOLDEN_DIR / "DOC-004_Disaster_Recovery_Runbook.pdf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the retrieval benchmark dataset JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_PATH,
        help="Path to write the benchmark results JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Loading benchmark queries...")
    queries = load_benchmark_queries(args.dataset)

    print("Loading embedding model...")
    embedding_provider = SentenceTransformerProvider()

    print("Creating retrieval pipeline...")
    pipeline = KnowledgePipeline(
        embedding_provider=embedding_provider,
        max_chunk_size=500,
    )

    print("Ingesting golden documents...")
    documents = [
        ingest_pdf(path, document_id)
        for path, document_id in zip(
            DOCUMENT_PATHS,
            ["DOC-001", "DOC-002", "DOC-003", "DOC-004"],
            strict=True,
        )
    ]

    print("Indexing documents...")
    pipeline.index_documents(documents)

    def retriever(query: str, top_k: int) -> list[str]:
        results = pipeline.search(
            query=query,
            top_k=top_k,
        )
        return [result.chunk.chunk_id for result in results]

    print("Running dense retrieval benchmark...")

    result = run_retrieval_benchmark(
        queries=queries,
        retriever=retriever,
        top_k=5,
    )

    output = {
        "retrieval_strategy": "dense",
        "embedding_model": embedding_provider.model_name,
        "chunk_size": 500,
        "top_k": 5,
        "query_count": result.query_count,
        "recall_at_5": result.recall_at_k,
        "precision_at_5": result.precision_at_k,
        "mrr": result.mrr,
        "ndcg_at_5": result.ndcg_at_k,
    }

    print("\n=== Dense Retrieval Baseline ===")
    print(json.dumps(output, indent=2))

    print("\n=== Per-Query Results ===")

    for query_result in result.query_results:
        print(
            f"{query_result.query_id}: "
            f"Recall@5={query_result.recall_at_k:.2f}, "
            f"Precision@5={query_result.precision_at_k:.2f}, "
            f"RR={query_result.reciprocal_rank:.2f}, "
            f"NDCG@5={query_result.ndcg_at_k:.2f}"
        )

        print(f"  Relevant:  {sorted(query_result.relevant_chunk_ids)}")
        print(f"  Retrieved: {query_result.retrieved_chunk_ids}")

    output["query_results"] = [
        {
            "query_id": query_result.query_id,
            "relevant_chunk_ids": sorted(query_result.relevant_chunk_ids),
            "retrieved_chunk_ids": query_result.retrieved_chunk_ids,
            "recall_at_5": query_result.recall_at_k,
            "precision_at_5": query_result.precision_at_k,
            "reciprocal_rank": query_result.reciprocal_rank,
            "ndcg_at_5": query_result.ndcg_at_k,
        }
        for query_result in result.query_results
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
