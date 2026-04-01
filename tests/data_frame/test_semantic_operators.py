from enum import Enum

import pytest

from evergreen.catalog.schema import Field, Schema
from evergreen.core.session_context import SessionContext
from evergreen.planner.logical.expr import prompt
from evergreen.storage.row import Row


@pytest.fixture
def sample_schema() -> Schema:
    return Schema((Field("id", int), Field("review", str)), ("id",))


@pytest.fixture
def sample_rows() -> list[Row]:
    return [
        Row((1, "This product sucks!")),
        Row((2, "I love this product!")),
        Row((3, "This product is okay.")),
    ]


def test_semantic_filter(
    ctx_with_model: SessionContext, sample_rows: list[Row], sample_schema: Schema
):
    ctx_with_model.enable_batching(batch_size=2)
    df = ctx_with_model.read_rows(sample_rows, sample_schema)
    result = df.filter(prompt("The {review} has positive sentiment.")).collect()

    assert result.rows == [
        Row((2, "I love this product!")),
    ]


def test_semantic_map(
    ctx_with_model: SessionContext, sample_rows: list[Row], sample_schema: Schema
):
    class Sentiment(Enum):
        POSITIVE = "positive"
        NEGATIVE = "negative"
        NEUTRAL = "neutral"

    ctx_with_model.enable_batching(batch_size=2)
    df = ctx_with_model.read_rows(sample_rows, sample_schema)
    df = df.map(
        prompt(
            "Classify the {review}'s sentiment as positive, negative, or neutral.",
            Sentiment,
        ).alias("sentiment")
    )
    result = df.collect()

    assert df.schema() == Schema(
        (
            Field("id", int),
            Field("review", str),
            Field("sentiment", Sentiment),
        ),
        ("id",),
    )
    assert result.rows == [
        Row((1, "This product sucks!", Sentiment.NEGATIVE)),
        Row((2, "I love this product!", Sentiment.POSITIVE)),
        Row((3, "This product is okay.", Sentiment.NEUTRAL)),
    ]
