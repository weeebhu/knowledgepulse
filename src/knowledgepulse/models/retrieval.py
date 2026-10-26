from typing import Literal

from pydantic import BaseModel, Field

from knowledgepulse.models.embedding import EmbeddedChunk


class RetrievalResult(BaseModel):
    """A retrieved chunk and its score, with the score's meaning made explicit."""

    chunk: EmbeddedChunk
    score: float = Field(ge=-1.0, le=1.0)
    score_type: Literal["cosine_similarity", "reciprocal_rank_fusion"] = "cosine_similarity"
