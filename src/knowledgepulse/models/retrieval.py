from pydantic import BaseModel, Field

from knowledgepulse.models.embedding import EmbeddedChunk


class RetrievalResult(BaseModel):
    """A retrieved chunk together with its similarity score."""

    chunk: EmbeddedChunk
    score: float = Field(ge=-1.0, le=1.0)
