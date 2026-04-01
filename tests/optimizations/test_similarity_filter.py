import pytest

from evergreen.catalog.schema import Field, Schema
from evergreen.core.session_context import SessionContext
from evergreen.planner.logical.expr import (
    col,
    prompt,
    proportion,
)
from evergreen.provenance import Prov
from evergreen.storage.row import Row
from tests.helpers import add_embeddings, validate_check_result


@pytest.mark.parametrize(
    "with_fusion,prov,prompt_count",
    [
        pytest.param(
            False,
            (
                Prov.pos_token((2,), col("is_best"))
                * Prov.pos_token((7,), col("is_best"))
            )
            + (
                Prov.pos_token((2,), col("is_best"))
                * Prov.pos_token((9,), col("is_best"))
            )
            + (
                Prov.pos_token((7,), col("is_best"))
                * Prov.pos_token((9,), col("is_best"))
            ),
            6 + 6,  # 6 tuples should make it past the similarity filter
            id="without_fusion",
        ),
        pytest.param(
            True,
            Prov.pos_token((2,), col("is_best")) * Prov.pos_token((7,), col("is_best")),
            6,  # 6 tuples should make it past the similarity filter
            id="with_fusion",
        ),
    ],
)
def test_similarity_filter(
    ctx_with_similarity_filter: SessionContext,
    with_fusion: bool,
    prov: Prov,
    prompt_count: int,
):
    if with_fusion:
        ctx_with_similarity_filter.enable_fusion()
        ctx_with_similarity_filter.enable_early_stop()
        ctx_with_similarity_filter.enable_minimal_provenance()

    schema = Schema((Field("id", int), Field("review", str)), ("id",))
    rows = [
        Row((1, "I had a terrible experience parking. I had to wait 30 minutes.")),
        Row((2, "The fish and chips were to die for! Best in town.")),
        Row((3, "The fish and chips were okay. Not the best in town.")),
        Row((4, "I was able to take lots of pretty pictures here.")),
        Row((5, "We drove 5 hours to come here and it was good.")),
        Row((6, "The fish and chips were awful. The place next door is better.")),
        Row((7, "They have the best fish and chips in Providence.")),
        Row((8, "The fish and chips were good. But the burger was better.")),
        Row((9, "Their fish and chips are the best around--super crispy.")),
        Row((10, "They charged me $100 for the parking... it was ridiculous.")),
    ]
    rows, schema = add_embeddings("review", rows, schema)
    df = ctx_with_similarity_filter.read_rows(rows, schema)
    result = (
        df.filter(prompt("the {review} mentions the restaurant's fish and chips"))
        .map(
            prompt(
                "identify whether the {review} calls the fish and chips the best in "
                "town"
            ).alias("is_best")
        )
        .aggregate([proportion(col("is_best")).alias("best_prop")])
        .check(col("best_prop") >= 0.2)
        .collect()
    )

    validate_check_result(result, True, prov)
    metrics = result.metrics.language_model_metrics
    assert len(metrics) == 1
    assert metrics[0].prompt_count == prompt_count
