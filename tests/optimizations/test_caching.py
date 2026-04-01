from evergreen.catalog.schema import Field, Schema
from evergreen.core.session_context import SessionContext
from evergreen.planner.logical.expr import col, count_if, prompt
from evergreen.storage.row import Row


def test_caching(ctx_with_caching: SessionContext):
    ctx_with_caching.enable_batching(batch_size=3)
    schema = Schema(
        (Field("id", int), Field("restaurant", str), Field("review", str)), ("id",)
    )
    rows = [
        Row(
            (1, "Restaurant A", "The restaurant had a fishy smell. Kinda like a sewer!")
        ),
        Row((2, "Restaurant A", "The fish and chips were to die for! Best in town.")),
        Row((3, "Restaurant A", "The fish and chips were okay. Not the best in town.")),
        Row(
            (
                4,
                "Restaurant B",
                "I order a burger and the fish and chips. It was amazing!",
            )
        ),
        Row((5, "Restaurant B", "I ordered the tuna melt. It was the best in town.")),
        Row(
            (
                6,
                "Restaurant B",
                "The fish and chips were awful. The place next door is better.",
            )
        ),
    ]
    df = ctx_with_caching.read_rows(rows, schema)

    result_1 = (
        df.map(
            prompt("identify whether the {review} has positive sentiment").alias(
                "is_positive"
            )
        )
        .aggregate(
            [count_if(col("is_positive")).alias("num_positive")],
            group_by=[col("restaurant")],
        )
        .with_rank(col("num_positive"))
        .filter(col("restaurant").eq("Restaurant A"))
        .check(col("rank").eq(2))
        .collect()
    )

    assert result_1.rows == [Row(("Restaurant A", True))]
    metrics = result_1.metrics.language_model_metrics
    assert len(metrics) == 1
    assert metrics[0].prompt_count == 6
    assert metrics[0].prompt_cache_hit_count == 0

    result_2 = (
        df.map(
            prompt("identify whether the {review} has positive sentiment").alias(
                "is_positive"
            )
        )
        .aggregate(
            [count_if(col("is_positive")).alias("num_positive")],
            group_by=[col("restaurant")],
        )
        .with_rank(col("num_positive"))
        .filter(col("restaurant").eq("Restaurant B"))
        .check(col("rank").eq(1))
        .collect()
    )

    assert result_2.rows == [Row(("Restaurant B", True))]
    metrics = result_2.metrics.language_model_metrics
    assert len(metrics) == 1
    assert metrics[0].prompt_count == 6
    assert metrics[0].prompt_cache_hit_count == 6
