import json
from pathlib import Path

from knowledgepulse.models.benchmark import BenchmarkQuery


def load_benchmark_queries(
    dataset_path: Path,
) -> list[BenchmarkQuery]:
    """Load and validate retrieval benchmark queries from JSON."""

    with dataset_path.open(encoding="utf-8") as file:
        raw_queries = json.load(file)

    return [BenchmarkQuery.model_validate(query) for query in raw_queries]
