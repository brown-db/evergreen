import pickle
from enum import Enum
from pathlib import Path

import pytest

from evergreen.catalog.schema import Field, Schema
from evergreen.core.session_context import SessionContext
from evergreen.planner.logical.expr import (
    bool_and,
    bool_or,
    col,
    count_if,
    proportion,
)
from evergreen.storage.row import Row


@pytest.fixture
def sample_schema() -> Schema:
    return Schema(
        (
            Field("id", int),
            Field("name", str),
            Field("age", int),
        ),
        ("id",),
    )


@pytest.fixture
def sample_rows() -> list[Row]:
    return [
        Row((1, "Alice", 30)),
        Row((2, "Bob", 25)),
        Row((3, "Charlie", 35)),
    ]


def test_schema(
    default_ctx: SessionContext, sample_rows: list[Row], sample_schema: Schema
):
    df = default_ctx.read_rows(sample_rows, sample_schema)

    assert df.schema() == sample_schema


def test_collect_all_rows(
    default_ctx: SessionContext, sample_rows: list[Row], sample_schema: Schema
):
    df = default_ctx.read_rows(sample_rows, sample_schema)
    result = df.collect()

    assert result.rows == sample_rows


def test_filter(
    default_ctx: SessionContext, sample_rows: list[Row], sample_schema: Schema
):
    df = default_ctx.read_rows(sample_rows, sample_schema)
    df = df.filter(col("age") > 28)
    result = df.collect()

    assert df.schema() == Schema(
        (
            Field("id", int),
            Field("name", str),
            Field("age", int),
        ),
        ("id",),
    )
    assert result.rows == [
        Row((1, "Alice", 30)),
        Row((3, "Charlie", 35)),
    ]


def test_map(
    default_ctx: SessionContext, sample_rows: list[Row], sample_schema: Schema
):
    df = default_ctx.read_rows(sample_rows, sample_schema)
    df = df.map((col("age") > 21).alias("is_adult"))
    result = df.collect()

    assert df.schema() == Schema(
        (
            Field("id", int),
            Field("name", str),
            Field("age", int),
            Field("is_adult", bool),
        ),
        ("id",),
    )
    assert result.rows == [
        Row((1, "Alice", 30, True)),
        Row((2, "Bob", 25, True)),
        Row((3, "Charlie", 35, True)),
    ]


def test_chained_operations(
    default_ctx: SessionContext, sample_rows: list[Row], sample_schema: Schema
):
    df = default_ctx.read_rows(sample_rows, sample_schema)
    df = df.filter((col("age") > 25) & (col("age") < 32)).map(
        (col("age").eq(30)).alias("is_thirty")
    )
    result = df.collect()

    assert df.schema() == Schema(
        (
            Field("id", int),
            Field("name", str),
            Field("age", int),
            Field("is_thirty", bool),
        ),
        ("id",),
    )
    assert result.rows == [
        Row((1, "Alice", 30, True)),
    ]


def test_map_field_name(
    default_ctx: SessionContext, sample_rows: list[Row], sample_schema: Schema
):
    df = default_ctx.read_rows(sample_rows, sample_schema)
    df = df.map(col("age").eq(30))
    result = df.collect()

    assert df.schema() == Schema(
        (
            Field("id", int),
            Field("name", str),
            Field("age", int),
            Field("age == 30", bool),
        ),
        ("id",),
    )
    assert result.rows == [
        Row((1, "Alice", 30, True)),
        Row((2, "Bob", 25, False)),
        Row((3, "Charlie", 35, False)),
    ]


def test_count_if(
    default_ctx: SessionContext, sample_rows: list[Row], sample_schema: Schema
):
    df = default_ctx.read_rows(sample_rows, sample_schema)
    df = df.aggregate(
        [count_if(col("age") > 28), count_if(col("age") <= 28).alias("le_28")]
    )
    result = df.collect()

    assert df.schema() == Schema(
        (
            Field(Row.INTERNAL_ID_FIELD_NAME, str),
            Field("count_if(age > 28)", int),
            Field("le_28", int),
        ),
        (Row.INTERNAL_ID_FIELD_NAME,),
    )
    assert result.rows == [Row((Row.AGG_ROW_ID, 2, 1))]


def test_proportion(
    default_ctx: SessionContext, sample_rows: list[Row], sample_schema: Schema
):
    df = default_ctx.read_rows(sample_rows, sample_schema)
    df = df.aggregate(
        [
            proportion(col("age") > 28),
            proportion(col("age") <= 28).alias("le_28"),
        ]
    )
    result = df.collect()

    assert df.schema() == Schema(
        (
            Field(Row.INTERNAL_ID_FIELD_NAME, str),
            Field("proportion(age > 28)", float),
            Field("le_28", float),
        ),
        (Row.INTERNAL_ID_FIELD_NAME,),
    )
    assert result.rows == [Row((Row.AGG_ROW_ID, 2 / 3, 1 / 3))]


def test_bool_and_true(default_ctx: SessionContext):
    schema = Schema((Field("id", int), Field("flag", bool)), ("id",))
    rows = [Row((1, True)), Row((2, True)), Row((3, True))]

    df = default_ctx.read_rows(rows, schema)
    result = df.aggregate([bool_and(col("flag"))]).collect()

    assert result.rows == [Row((Row.AGG_ROW_ID, True))]


def test_bool_and_false(default_ctx: SessionContext):
    schema = Schema((Field("id", int), Field("flag", bool)), ("id",))
    rows = [Row((1, True)), Row((2, False)), Row((3, True))]

    df = default_ctx.read_rows(rows, schema)
    result = df.aggregate([bool_and(col("flag"))]).collect()

    assert result.rows == [Row((Row.AGG_ROW_ID, False))]


def test_bool_or_true(default_ctx: SessionContext):
    schema = Schema((Field("id", int), Field("flag", bool)), ("id",))
    rows = [Row((1, False)), Row((2, False)), Row((3, True))]

    df = default_ctx.read_rows(rows, schema)
    result = df.aggregate([bool_or(col("flag"))]).collect()

    assert result.rows == [Row((Row.AGG_ROW_ID, True))]


def test_bool_or_false(default_ctx: SessionContext):
    schema = Schema((Field("id", int), Field("flag", bool)), ("id",))
    rows = [Row((1, False)), Row((2, False)), Row((3, False))]

    df = default_ctx.read_rows(rows, schema)
    result = df.aggregate([bool_or(col("flag"))]).collect()

    assert result.rows == [Row((Row.AGG_ROW_ID, False))]


def test_group_by_aggregate(default_ctx: SessionContext):
    schema = Schema(
        (Field("id", int), Field("group", str), Field("flag", bool)), ("id",)
    )
    rows = [
        Row((1, "A", True)),
        Row((2, "B", False)),
        Row((3, "A", False)),
        Row((4, "C", True)),
        Row((5, "B", False)),
    ]
    df = default_ctx.read_rows(rows, schema)
    result = df.aggregate([count_if(col("flag"))], group_by=[col("group")]).collect()

    assert result.rows == [Row(("A", 1)), Row(("B", 0)), Row(("C", 1))]


def test_nested_group_by_aggregate(default_ctx: SessionContext):
    schema = Schema(
        (
            Field("id", int),
            Field("group1", str),
            Field("group2", int),
            Field("flag", bool),
        ),
        ("id",),
    )
    rows = [
        Row((1, "A", 1, True)),
        Row((2, "B", 1, False)),
        Row((3, "A", 1, False)),
        Row((4, "C", 3, True)),
        Row((3, "A", 2, False)),
        Row((5, "B", 2, False)),
    ]
    df = default_ctx.read_rows(rows, schema)
    result = df.aggregate(
        [count_if(col("flag"))], group_by=[col("group1"), col("group2")]
    ).collect()

    assert result.rows == [
        Row(("A", 1, 1)),
        Row(("A", 2, 0)),
        Row(("B", 1, 0)),
        Row(("B", 2, 0)),
        Row(("C", 3, 1)),
    ]


def test_check(
    default_ctx: SessionContext, sample_rows: list[Row], sample_schema: Schema
):
    df = default_ctx.read_rows(sample_rows, sample_schema)

    assert df.aggregate([count_if(col("age") > 28).alias("over_28")]).check(
        col("over_28").eq(2)
    ).collect().rows == [Row((Row.AGG_ROW_ID, True))]

    assert df.aggregate([count_if(col("age") > 28).alias("over_28")]).check(
        ~(col("over_28").eq(2))
    ).collect().rows == [Row((Row.AGG_ROW_ID, False))]


def test_check_argmax(
    default_ctx: SessionContext, sample_rows: list[Row], sample_schema: Schema
):
    df = default_ctx.read_rows(sample_rows, sample_schema)
    result = (
        df.with_rank(col("age"))
        .filter(col("name").eq("Charlie"))
        .check(col("rank").eq(1))
        .collect()
    )

    assert result.rows == [Row((3, True))]


def test_check_argmin(
    default_ctx: SessionContext, sample_rows: list[Row], sample_schema: Schema
):
    df = default_ctx.read_rows(sample_rows, sample_schema)
    result = (
        df.with_rank(col("age"), descending=False)
        .filter(col("name").eq("Bob"))
        .check(col("rank").eq(1))
        .collect()
    )

    assert result.rows == [Row((2, True))]


def test_check_argmax_with_ties(default_ctx: SessionContext):
    schema = Schema(
        (
            Field("id", int),
            Field("name", str),
            Field("score", int),
        ),
        ("id",),
    )
    # Alice and Bob tie for #1 (score=95), Charlie is #2 (score=78)
    rows = [
        Row((1, "Alice", 95)),
        Row((2, "Bob", 95)),
        Row((3, "Charlie", 78)),
    ]

    df = default_ctx.read_rows(rows, schema)

    # Alice is #1 (tied)
    result = (
        df.with_rank(col("score"))
        .filter(col("name").eq("Alice"))
        .check(col("rank").eq(1))
        .collect()
    )
    assert result.rows == [Row((1, True))]

    # Bob is also #1 (tied)
    result = (
        df.with_rank(col("score"))
        .filter(col("name").eq("Bob"))
        .check(col("rank").eq(1))
        .collect()
    )
    assert result.rows == [Row((2, True))]

    # Charlie is #2 (dense rank - no gap)
    result = (
        df.with_rank(col("score"))
        .filter(col("name").eq("Charlie"))
        .check(col("rank").eq(2))
        .collect()
    )
    assert result.rows == [Row((3, True))]

    # Charlie is NOT #3 (would be #3 with RANK, but #2 with DENSE_RANK)
    result = (
        df.with_rank(col("score"))
        .filter(col("name").eq("Charlie"))
        .check(col("rank").eq(3))
        .collect()
    )
    assert result.rows == [Row((3, False))]


def test_check_with_compound_predicate(default_ctx: SessionContext):
    class Sentiment(Enum):
        POSITIVE = "positive"
        NEGATIVE = "negative"
        NEUTRAL = "neutral"

    schema = Schema((Field("id", int), Field("sentiment", Sentiment)), ("id",))
    rows = [
        Row((1, Sentiment.POSITIVE)),
        Row((2, Sentiment.NEGATIVE)),
        Row((3, Sentiment.NEUTRAL)),
        Row((4, Sentiment.POSITIVE)),
        Row((5, Sentiment.POSITIVE)),
        Row((6, Sentiment.NEGATIVE)),
        Row((7, Sentiment.NEUTRAL)),
        Row((8, Sentiment.POSITIVE)),
        Row((9, Sentiment.NEGATIVE)),
        Row((10, Sentiment.NEGATIVE)),
    ]

    df = default_ctx.read_rows(rows, schema)
    result = (
        df.aggregate(
            [
                proportion(col("sentiment").eq(Sentiment.POSITIVE)).alias(
                    "positive_prop"
                ),
                proportion(col("sentiment").eq(Sentiment.NEGATIVE)).alias(
                    "negative_prop"
                ),
            ]
        )
        .check((col("positive_prop") >= 0.4) & (col("negative_prop") >= 0.4))
        .collect()
    )
    assert result.rows == [Row((Row.AGG_ROW_ID, True))]


def test_log(
    default_ctx: SessionContext,
    sample_rows: list[Row],
    sample_schema: Schema,
    tmp_path: Path,
):
    pickle_path = tmp_path / "log.pkl"

    df = default_ctx.read_rows(sample_rows, sample_schema)
    result = (
        df.filter(col("age") > 28)
        .log(str(pickle_path))
        .map(col("age").eq(30).alias("is_thirty"))
        .collect()
    )

    assert len(result.rows) == 2

    with open(pickle_path, "rb") as f:
        logged_rows, logged_schema = pickle.load(f)

    assert logged_rows == [
        Row((1, "Alice", 30)),
        Row((3, "Charlie", 35)),
    ]
    assert logged_schema == sample_schema
