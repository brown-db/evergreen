from enum import Enum

from evergreen.catalog.schema import Field, Schema
from evergreen.core.session_context import SessionContext
from evergreen.planner.logical.expr import bool_and, bool_or, col, count_if, proportion
from evergreen.provenance import Prov
from evergreen.storage.row import Row
from tests.helpers import validate_check_result


def test_nested_quantifiers_with_cardinal(ctx_with_early_stop: SessionContext):
    class Sentiment(Enum):
        POSITIVE = "positive"
        NEGATIVE = "negative"
        NEUTRAL = "neutral"

    schema = Schema(
        (
            Field("id", int),
            Field("restaurant", str),
            Field("dish", str),
            Field("sentiment", Sentiment),
        ),
        ("id",),
    )
    rows = [
        Row((1, "Restaurant A", "Pizza", Sentiment.NEGATIVE)),
        Row((2, "Restaurant A", "Salad", Sentiment.POSITIVE)),
        Row((3, "Restaurant A", "Pizza", Sentiment.POSITIVE)),
        Row((4, "Restaurant A", "Salad", Sentiment.POSITIVE)),
        Row((5, "Restaurant A", "Pizza", Sentiment.POSITIVE)),
        Row((6, "Restaurant A", "Pizza", Sentiment.POSITIVE)),
        Row((7, "Restaurant A", "Salad", Sentiment.POSITIVE)),
        Row((8, "Restaurant B", "Pasta", Sentiment.POSITIVE)),
        Row((9, "Restaurant B", "Pasta", Sentiment.POSITIVE)),
        Row((10, "Restaurant B", "Salad", Sentiment.NEUTRAL)),
        Row((11, "Restaurant B", "Pasta", Sentiment.POSITIVE)),
    ]
    df = ctx_with_early_stop.read_rows(rows, schema)

    # For all restaurants, there exists a dish with more than 2 positive sentiments
    result = (
        df.aggregate(
            [count_if(col("sentiment").eq(Sentiment.POSITIVE)).alias("num_positive")],
            group_by=[col("restaurant"), col("dish")],
        )
        .aggregate(
            [bool_or(col("num_positive") > 2).alias("exists_popular_dish")],
            group_by=[col("restaurant")],
        )
        .aggregate(
            [
                bool_and(col("exists_popular_dish")).alias(
                    "all_restaurants_have_popular_dish"
                )
            ]
        )
        .check(col("all_restaurants_have_popular_dish"))
        .collect()
    )
    prov = (
        Prov.pos_token((3,), col("sentiment").eq(Sentiment.POSITIVE))
        * Prov.pos_token((5,), col("sentiment").eq(Sentiment.POSITIVE))
        * Prov.pos_token((6,), col("sentiment").eq(Sentiment.POSITIVE))
    ) * (
        Prov.pos_token((8,), col("sentiment").eq(Sentiment.POSITIVE))
        * Prov.pos_token((9,), col("sentiment").eq(Sentiment.POSITIVE))
        * Prov.pos_token((11,), col("sentiment").eq(Sentiment.POSITIVE))
    )

    validate_check_result(result, True, prov)


def test_nested_quantifiers_with_proportion(ctx_with_early_stop: SessionContext):
    class Sentiment(Enum):
        POSITIVE = "positive"
        NEGATIVE = "negative"
        NEUTRAL = "neutral"

    schema = Schema(
        (
            Field("id", int),
            Field("restaurant", str),
            Field("dish", str),
            Field("sentiment", Sentiment),
        ),
        ("id",),
    )
    rows = [
        Row((1, "Restaurant A", "Pizza", Sentiment.NEGATIVE)),
        Row((2, "Restaurant A", "Salad", Sentiment.POSITIVE)),
        Row((3, "Restaurant A", "Pizza", Sentiment.POSITIVE)),
        Row((4, "Restaurant A", "Salad", Sentiment.POSITIVE)),
        Row((5, "Restaurant A", "Pizza", Sentiment.POSITIVE)),
        Row((6, "Restaurant A", "Pizza", Sentiment.POSITIVE)),
        Row((7, "Restaurant A", "Salad", Sentiment.POSITIVE)),
        Row((8, "Restaurant B", "Pasta", Sentiment.POSITIVE)),
        Row((9, "Restaurant B", "Pasta", Sentiment.POSITIVE)),
        Row((10, "Restaurant B", "Salad", Sentiment.NEUTRAL)),
        Row((11, "Restaurant B", "Pasta", Sentiment.POSITIVE)),
    ]
    df = ctx_with_early_stop.read_rows(rows, schema)

    # For all restaurants, there exists a dish with more than 50% positive sentiments
    result = (
        df.aggregate(
            [
                proportion(col("sentiment").eq(Sentiment.POSITIVE)).alias(
                    "prop_positive"
                )
            ],
            group_by=[col("restaurant"), col("dish")],
        )
        .aggregate(
            [bool_or(col("prop_positive") > 0.5).alias("exists_popular_dish")],
            group_by=[col("restaurant")],
        )
        .aggregate(
            [
                bool_and(col("exists_popular_dish")).alias(
                    "all_restaurants_have_popular_dish"
                )
            ]
        )
        .check(col("all_restaurants_have_popular_dish"))
        .collect()
    )
    prov = (
        Prov.pos_token((3,), col("sentiment").eq(Sentiment.POSITIVE))
        * Prov.pos_token((5,), col("sentiment").eq(Sentiment.POSITIVE))
        * Prov.pos_token((6,), col("sentiment").eq(Sentiment.POSITIVE))
    ) * (
        Prov.pos_token((8,), col("sentiment").eq(Sentiment.POSITIVE))
        * Prov.pos_token((9,), col("sentiment").eq(Sentiment.POSITIVE))
    )

    validate_check_result(result, True, prov)
