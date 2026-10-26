import argparse
import json
from pathlib import Path

from knowledgepulse.benchmarks.dataset import load_benchmark_queries
from knowledgepulse.embeddings.sentence_transformer import SentenceTransformerProvider
from knowledgepulse.embeddings.service import embed_chunks
from knowledgepulse.evaluation.retrieval_benchmark import run_retrieval_benchmark
from knowledgepulse.ingestion.chunking import chunk_document
from knowledgepulse.ingestion.pdf import ingest_pdf
from knowledgepulse.retrieval.in_memory_vector_store import InMemoryVectorStore

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
DOCUMENT_IDS = ["DOC-001", "DOC-002", "DOC-003", "DOC-004"]
CHUNK_SIZE = 500
TOP_K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=RESULTS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queries = load_benchmark_queries(args.dataset)

    print("Loading embedding model...")
    embedding_provider = SentenceTransformerProvider()

    print("Ingesting and chunking golden documents...")
    documents = [
        ingest_pdf(path, document_id)
        for path, document_id in zip(DOCUMENT_PATHS, DOCUMENT_IDS, strict=True)
    ]
    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(document, max_chunk_size=CHUNK_SIZE)
    ]

    print(f"Embedding {len(chunks)} chunks for dense retrieval...")
    embedded_chunks = embed_chunks(chunks, embedding_provider)
    vector_store = InMemoryVectorStore()
    vector_store.add(embedded_chunks)

    def dense_retrieve(query: str, top_k: int) -> list[str]:
        query_vector = embedding_provider.embed_many([query])[0]
        results = vector_store.search(query_vector, top_k)
        return [result.chunk.chunk_id for result in results]

    print("Running dense-only retrieval benchmark...")
    result = run_retrieval_benchmark(queries, dense_retrieve, TOP_K)

    output = {
        "retrieval_strategy": "dense",
        "embedding_model": embedding_provider.model_name,
        "chunk_size": CHUNK_SIZE,
        "top_k": TOP_K,
        "indexed_chunk_count": len(embedded_chunks),
        "query_count": result.query_count,
        "recall_at_5": result.recall_at_k,
        "precision_at_5": result.precision_at_k,
        "mrr": result.mrr,
        "ndcg_at_5": result.ndcg_at_k,
        "query_results": [
            {
                "query_id": item.query_id,
                "relevant_chunk_ids": sorted(item.relevant_chunk_ids),
                "retrieved_chunk_ids": item.retrieved_chunk_ids,
                "recall_at_5": item.recall_at_k,
                "precision_at_5": item.precision_at_k,
                "reciprocal_rank": item.reciprocal_rank,
                "ndcg_at_5": item.ndcg_at_k,
            }
            for item in result.query_results
        ],
    }

    print("\n=== Dense Retrieval Baseline ===")
    print(json.dumps({key: value for key, value in output.items() if key != "query_results"}, indent=2))
    for item in result.query_results:
        print(
            f"{item.query_id}: Recall@5={item.recall_at_k:.2f}, "
            f"Precision@5={item.precision_at_k:.2f}, "
            f"RR={item.reciprocal_rank:.2f}, NDCG@5={item.ndcg_at_k:.2f}"
        )
        print(f"  Relevant:  {sorted(item.relevant_chunk_ids)}")
        print(f"  Retrieved: {item.retrieved_chunk_ids}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
