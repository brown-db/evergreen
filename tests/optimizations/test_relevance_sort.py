from enum import Enum

from evergreen.catalog.schema import Field, Schema
from evergreen.core.session_context import SessionContext
from evergreen.planner.logical.expr import (
    bool_or,
    col,
    count_if,
    prompt,
    proportion,
)
from evergreen.provenance import Prov
from evergreen.storage.row import Row
from tests.helpers import add_embeddings, validate_check_result


def test_existential_claim(ctx_with_relevance_sort: SessionContext):
    schema = Schema((Field("id", int), Field("review", str)), ("id",))
    rows = [
        Row((1, "The ice cream was delicious!")),
        Row((2, "The service was fast. I only waited for 1 minute.")),
        Row((3, "I really enjoyed the atmosphere.")),
        Row((4, "The burger was burnt.")),
        Row((5, "The service was slow. Waited 30 minutes for the server.")),
    ]
    rows, schema = add_embeddings("review", rows, schema)
    df = ctx_with_relevance_sort.read_rows(rows, schema)
    result = (
        df.map(
            prompt(
                "identify whether the {review} mentions waiting 20+ minutes to be "
                "acknowledged",
                bool,
            ).alias("is_long_wait")
        )
        .aggregate([bool_or(col("is_long_wait")).alias("exists_long_wait")])
        .check(col("exists_long_wait"))
        .collect()
    )
    prov = Prov.pos_token((5,), col("is_long_wait"))

    validate_check_result(result, True, prov)

    language_model_metrics = result.metrics.language_model_metrics
    assert len(language_model_metrics) == 1
    assert language_model_metrics[0].prompt_count == 1

    optimizer_language_model_metrics = result.metrics.optimizer_language_model_metrics
    assert len(optimizer_language_model_metrics) == 1
    assert optimizer_language_model_metrics[0].prompt_count == 1


def test_cardinal_claim(ctx_with_relevance_sort: SessionContext):
    schema = Schema((Field("id", int), Field("review", str)), ("id",))
    rows = [
        Row((1, "The restaurant had a weird smell. Kinda like a sewer!")),
        Row((2, "The service was fast. I only waited for 1 minute.")),
        Row((3, "I had to wait so long (~20 min) to get the server's attention.")),
        Row((4, "The burger was burnt.")),
        Row((5, "It smelled terrible. I had to leave. Like poop!")),
    ]
    rows, schema = add_embeddings("review", rows, schema)
    df = ctx_with_relevance_sort.read_rows(rows, schema)
    result = (
        df.map(
            prompt("identify whether the {review} mentions sewer smell problems").alias(
                "has_sewer_problems"
            )
        )
        .aggregate([count_if(col("has_sewer_problems")).alias("num_sewer_problems")])
        .check(col("num_sewer_problems") >= 2)
        .collect()
    )
    prov = Prov.pos_token((1,), col("has_sewer_problems")) * Prov.pos_token(
        (5,), col("has_sewer_problems")
    )

    validate_check_result(result, True, prov)

    language_model_metrics = result.metrics.language_model_metrics
    assert len(language_model_metrics) == 1
    assert language_model_metrics[0].prompt_count == 2

    optimizer_language_model_metrics = result.metrics.optimizer_language_model_metrics
    assert len(optimizer_language_model_metrics) == 1
    assert optimizer_language_model_metrics[0].prompt_count == 1


def test_proportional_claim(ctx_with_relevance_sort: SessionContext):
    """Tests that we don't insert a relevance sort for proportional claims."""

    schema = Schema((Field("id", int), Field("review", str)), ("id",))
    rows = [
        Row((1, "The restaurant had a awful fishy smell. Kinda like a sewer!")),
        Row((2, "The food was mediocre.")),
        Row((3, "The food was terrible. I had to leave.")),
        Row((4, "The service was slow.")),
        Row((5, "The fish and chips were amazing! Best in town.")),
    ]
    rows, schema = add_embeddings("review", rows, schema)
    df = ctx_with_relevance_sort.read_rows(rows, schema)

    class Sentiment(Enum):
        POSITIVE = "positive"
        NEGATIVE = "negative"
        NEUTRAL = "neutral"

    result = (
        df.map(
            prompt("determine the sentiment of the {review}", Sentiment).alias(
                "sentiment"
            )
        )
        .aggregate(
            [proportion(col("sentiment").eq(Sentiment.POSITIVE)).alias("positive_prop")]
        )
        .check(col("positive_prop") >= 0.1)
        .collect()
    )
    prov = Prov.pos_token((5,), col("sentiment").eq(Sentiment.POSITIVE))

    validate_check_result(result, True, prov)

    language_model_metrics = result.metrics.language_model_metrics
    assert len(language_model_metrics) == 1
    assert language_model_metrics[0].prompt_count == 5

    optimizer_language_model_metrics = result.metrics.optimizer_language_model_metrics
    assert len(optimizer_language_model_metrics) == 1
    assert optimizer_language_model_metrics[0].prompt_count == 0
