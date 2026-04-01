from evergreen.catalog.schema import Field, Schema
from evergreen.core.session_context import SessionContext
from evergreen.planner.logical.expr import col, prompt, proportion
from evergreen.provenance import Prov
from evergreen.storage.row import Row
from tests.helpers import validate_check_result


def test_fusion(ctx_with_fusion: SessionContext):
    ctx_with_fusion.enable_batching(batch_size=5)
    schema = Schema((Field("id", int), Field("review", str)), ("id",))
    rows = [
        Row((1, "The restaurant had a fishy smell. Kinda like a sewer!")),
        Row((2, "The fish and chips were to die for! Best in town.")),
        Row((3, "The fish and chips were okay. Not the best in town.")),
        Row((4, "I order a burger and the fish and chips.")),
        Row((5, "I ordered the tuna melt. It was the best in town.")),
        Row((6, "The fish and chips were awful. The place next door is better.")),
        Row((7, "They have the best fish and chips in Providence.")),
        Row((8, "The fish and chips were good. But the burger was better.")),
        Row((9, "Their fish and chips are the best around--super crispy.")),
        Row((10, "Their salt and vinegar chips are the best in town.")),
    ]
    df = ctx_with_fusion.read_rows(rows, schema)
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
    prov = Prov.pos_token((2,), col("is_best")) * Prov.pos_token((7,), col("is_best"))

    validate_check_result(result, True, prov)
    metrics = result.metrics.language_model_metrics
    assert len(metrics) == 1
    assert metrics[0].prompt_count == 10
