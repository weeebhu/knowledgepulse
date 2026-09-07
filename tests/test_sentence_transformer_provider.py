from knowledgepulse.embeddings.sentence_transformer import (
    SentenceTransformerProvider,
)


def test_sentence_transformer_provider_embeds_text() -> None:
    """Provider should convert text into a real embedding vector."""

    provider = SentenceTransformerProvider()

    vectors = provider.embed_many(
        ["Production deployments require approval."],
    )

    assert len(vectors) == 1
    assert isinstance(vectors[0], list)
    assert len(vectors[0]) > 0
    assert all(isinstance(value, float) for value in vectors[0])


def test_sentence_transformer_provider_embeds_multiple_texts() -> None:
    """Provider should embed multiple texts in one batch."""

    provider = SentenceTransformerProvider()

    texts = [
        "Production deployments require approval.",
        "Database backups must be verified.",
    ]

    vectors = provider.embed_many(texts)

    assert len(vectors) == len(texts)
    assert all(isinstance(vector, list) for vector in vectors)
    assert all(len(vector) > 0 for vector in vectors)


def test_sentence_transformer_provider_returns_consistent_dimensions() -> None:
    """All vectors from the provider should have the same dimension."""

    provider = SentenceTransformerProvider()

    vectors = provider.embed_many(
        [
            "Production deployments require approval.",
            "Rollback procedures must be documented.",
        ],
    )

    dimensions = {len(vector) for vector in vectors}

    assert len(dimensions) == 1
