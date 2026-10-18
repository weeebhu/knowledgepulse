import math
import re

from knowledgepulse.models.chunk import TextChunk


class BM25Retriever:
    """A small BM25 lexical retriever over text chunks."""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")

        self.k1 = k1
        self.b = b
        self._chunks: list[TextChunk] = []
        self._tokens: list[list[str]] = []
        self._document_frequencies: dict[str, int] = {}
        self._average_length = 0.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Normalize text and extract alphanumeric tokens."""
        return re.findall(r"\b\w+\b", text.casefold())

    def index(self, chunks: list[TextChunk]) -> None:
        """Replace the current index with the supplied chunks."""
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Duplicate chunk ID")

        self._chunks = list(chunks)
        self._tokens = [self._tokenize(chunk.text) for chunk in chunks]
        self._document_frequencies = {}
        total_length = 0

        for tokens in self._tokens:
            total_length += len(tokens)

            for token in set(tokens):
                self._document_frequencies[token] = self._document_frequencies.get(token, 0) + 1

        self._average_length = total_length / len(self._tokens) if self._tokens else 0.0

    def search(self, query: str, top_k: int) -> list[TextChunk]:
        """Return chunks ranked by descending BM25 score."""
        if not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not self._chunks:
            return []

        query_tokens = set(self._tokenize(query))
        document_count = len(self._chunks)
        scored_chunks: list[tuple[int, float]] = []

        for index, tokens in enumerate(self._tokens):
            if not tokens:
                continue

            term_frequencies: dict[str, int] = {}
            for token in tokens:
                term_frequencies[token] = term_frequencies.get(token, 0) + 1

            score = 0.0
            document_length = len(tokens)

            for token in query_tokens:
                frequency = term_frequencies.get(token, 0)
                if frequency == 0:
                    continue

                document_frequency = self._document_frequencies[token]
                inverse_document_frequency = math.log(
                    1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
                )

                length_normalization = self.k1 * (
                    1 - self.b + self.b * document_length / self._average_length
                )

                score += (
                    inverse_document_frequency
                    * (frequency * (self.k1 + 1))
                    / (frequency + length_normalization)
                )

            if score > 0:
                scored_chunks.append((index, score))

        scored_chunks.sort(key=lambda item: -item[1])

        return [self._chunks[index] for index, _ in scored_chunks[:top_k]]
