import pytest

from evergreen.catalog.schema import Field, Schema
from evergreen.core.session_context import SessionContext
from evergreen.data_frame import FilterMetrics, MapMetrics
from evergreen.planner.logical.expr import col
from evergreen.storage.row import Row


@pytest.fixture
def sample_filter_schema() -> Schema:
    return Schema(
        (
            Field("id", int),
            Field("name", str),
            Field("age", int),
        ),
        ("id",),
    )


@pytest.fixture
def sample_filter_rows() -> list[Row]:
    return [
        Row((1, "Alice", 30)),
        Row((2, "Bob", 25)),
        Row((3, "Charlie", 35)),
        Row((4, "Diana", 28)),
        Row((5, "Eve", 40)),
    ]


def test_evaluate_filter_perfect(
    default_ctx: SessionContext,
    sample_filter_rows: list[Row],
    sample_filter_schema: Schema,
):
    reference_df = default_ctx.read_rows(
        sample_filter_rows, sample_filter_schema
    ).filter(col("age") > 28)
    pre_sem_op_df = default_ctx.read_rows(sample_filter_rows, sample_filter_schema)
    post_sem_op_df = default_ctx.read_rows(
        sample_filter_rows, sample_filter_schema
    ).filter(col("age") > 28)

    metrics = reference_df.evaluate_filter(pre_sem_op_df, post_sem_op_df)

    assert metrics.tp == 3
    assert metrics.fp == 0
    assert metrics.fn == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1_score == 1.0


def test_evaluate_filter_false_positives(
    default_ctx: SessionContext,
    sample_filter_schema: Schema,
):
    reference_rows = [Row((1, "Alice", 30))]
    reference_df = default_ctx.read_rows(reference_rows, sample_filter_schema)

    pre_rows = [Row((1, "Alice", 30)), Row((2, "Bob", 25))]
    pre_sem_op_df = default_ctx.read_rows(pre_rows, sample_filter_schema)

    post_rows = [Row((1, "Alice", 30)), Row((2, "Bob", 25))]
    post_sem_op_df = default_ctx.read_rows(post_rows, sample_filter_schema)

    metrics = reference_df.evaluate_filter(pre_sem_op_df, post_sem_op_df)

    assert metrics.tp == 1
    assert metrics.fp == 1
    assert metrics.fn == 0
    assert metrics.precision == 0.5
    assert metrics.recall == 1.0


def test_evaluate_filter_false_negatives(
    default_ctx: SessionContext,
    sample_filter_rows: list[Row],
    sample_filter_schema: Schema,
):
    reference_rows = [Row((1, "Alice", 30)), Row((3, "Charlie", 35))]
    reference_df = default_ctx.read_rows(reference_rows, sample_filter_schema)

    pre_sem_op_df = default_ctx.read_rows(sample_filter_rows, sample_filter_schema)

    post_rows = [Row((1, "Alice", 30))]
    post_sem_op_df = default_ctx.read_rows(post_rows, sample_filter_schema)

    metrics = reference_df.evaluate_filter(pre_sem_op_df, post_sem_op_df)

    assert metrics.tp == 1
    assert metrics.fp == 0
    assert metrics.fn == 1
    assert metrics.precision == 1.0
    assert metrics.recall == 0.5


def test_evaluate_filter_mixed_errors(
    default_ctx: SessionContext,
    sample_filter_rows: list[Row],
    sample_filter_schema: Schema,
):
    reference_rows = [Row((1, "Alice", 30)), Row((3, "Charlie", 35))]
    reference_df = default_ctx.read_rows(reference_rows, sample_filter_schema)

    pre_sem_op_df = default_ctx.read_rows(sample_filter_rows, sample_filter_schema)

    post_rows = [Row((1, "Alice", 30)), Row((2, "Bob", 25))]
    post_sem_op_df = default_ctx.read_rows(post_rows, sample_filter_schema)

    metrics = reference_df.evaluate_filter(pre_sem_op_df, post_sem_op_df)

    assert metrics.tp == 1
    assert metrics.fp == 1
    assert metrics.fn == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5


def test_evaluate_filter_empty_reference(
    default_ctx: SessionContext,
    sample_filter_rows: list[Row],
    sample_filter_schema: Schema,
):
    reference_df = default_ctx.read_rows([], sample_filter_schema)
    pre_sem_op_df = default_ctx.read_rows(sample_filter_rows, sample_filter_schema)

    post_rows = [Row((2, "Bob", 25))]
    post_sem_op_df = default_ctx.read_rows(post_rows, sample_filter_schema)

    metrics = reference_df.evaluate_filter(pre_sem_op_df, post_sem_op_df)

    assert metrics.tp == 0
    assert metrics.fp == 1
    assert metrics.fn == 0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0


def test_evaluate_filter_empty_output(
    default_ctx: SessionContext,
    sample_filter_rows: list[Row],
    sample_filter_schema: Schema,
):
    reference_rows = [Row((1, "Alice", 30)), Row((3, "Charlie", 35))]
    reference_df = default_ctx.read_rows(reference_rows, sample_filter_schema)

    pre_sem_op_df = default_ctx.read_rows(sample_filter_rows, sample_filter_schema)
    post_sem_op_df = default_ctx.read_rows([], sample_filter_schema)

    metrics = reference_df.evaluate_filter(pre_sem_op_df, post_sem_op_df)

    assert metrics.tp == 0
    assert metrics.fp == 0
    assert metrics.fn == 2
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0


@pytest.fixture
def sample_map_schema() -> Schema:
    return Schema(
        (Field("id", int), Field("name", str), Field("is_adult", bool)),
        ("id",),
    )


@pytest.fixture
def sample_map_rows() -> list[Row]:
    return [
        Row((1, "Alice", True)),
        Row((2, "Bob", True)),
        Row((3, "Charlie", False)),
    ]


def test_evaluate_map_all_correct(
    default_ctx: SessionContext, sample_map_rows: list[Row], sample_map_schema: Schema
):
    reference_df = default_ctx.read_rows(sample_map_rows, sample_map_schema)
    post_sem_op_df = default_ctx.read_rows(sample_map_rows, sample_map_schema)

    metrics = reference_df.evaluate_map(post_sem_op_df, [col("is_adult")])

    assert metrics.correct_by_column == {"is_adult": 3}
    assert metrics.total_rows == 3
    assert metrics.accuracy == 1.0


def test_evaluate_map_some_incorrect(
    default_ctx: SessionContext, sample_map_rows: list[Row], sample_map_schema: Schema
):
    reference_df = default_ctx.read_rows(sample_map_rows, sample_map_schema)

    post_rows = [
        Row((1, "Alice", True)),
        Row((2, "Bob", True)),
        Row((3, "Charlie", True)),
    ]
    post_sem_op_df = default_ctx.read_rows(post_rows, sample_map_schema)

    metrics = reference_df.evaluate_map(post_sem_op_df, [col("is_adult")])

    assert metrics.correct_by_column == {"is_adult": 2}
    assert metrics.total_rows == 3
    assert metrics.accuracy == 2 / 3


def test_evaluate_map_all_incorrect(
    default_ctx: SessionContext, sample_map_rows: list[Row], sample_map_schema: Schema
):
    reference_df = default_ctx.read_rows(sample_map_rows, sample_map_schema)

    post_rows = [
        Row((1, "Alice", False)),
        Row((2, "Bob", False)),
        Row((3, "Charlie", True)),
    ]
    post_sem_op_df = default_ctx.read_rows(post_rows, sample_map_schema)

    metrics = reference_df.evaluate_map(post_sem_op_df, [col("is_adult")])

    assert metrics.correct_by_column == {"is_adult": 0}
    assert metrics.total_rows == 3
    assert metrics.accuracy == 0.0


def test_evaluate_map_multiple_columns(default_ctx: SessionContext):
    schema = Schema(
        (Field("id", int), Field("is_adult", bool), Field("is_senior", bool)),
        ("id",),
    )
    reference_rows = [
        Row((1, True, False)),
        Row((2, True, True)),
        Row((3, False, False)),
    ]
    reference_df = default_ctx.read_rows(reference_rows, schema)

    post_rows = [
        Row((1, True, False)),
        Row((2, False, True)),
        Row((3, False, False)),
    ]
    post_sem_op_df = default_ctx.read_rows(post_rows, schema)

    metrics = reference_df.evaluate_map(
        post_sem_op_df, [col("is_adult"), col("is_senior")]
    )

    assert metrics.correct_by_column == {"is_adult": 2, "is_senior": 3}
    assert metrics.total_rows == 3
    assert metrics.accuracy == 5 / 6


def test_evaluate_map_empty(default_ctx: SessionContext, sample_map_schema: Schema):
    reference_df = default_ctx.read_rows([], sample_map_schema)
    post_sem_op_df = default_ctx.read_rows([], sample_map_schema)

    metrics = reference_df.evaluate_map(post_sem_op_df, [col("is_adult")])

    assert metrics.correct_by_column == {"is_adult": 0}
    assert metrics.total_rows == 0
    assert metrics.accuracy == 0.0


def test_filter_metrics_precision_zero_division():
    metrics = FilterMetrics(tp=0, fp=0, fn=5)
    assert metrics.precision == 0.0


def test_filter_metrics_recall_zero_division():
    metrics = FilterMetrics(tp=0, fp=5, fn=0)
    assert metrics.recall == 0.0


def test_filter_metrics_f1_zero_division():
    metrics = FilterMetrics(tp=0, fp=0, fn=0)
    assert metrics.f1_score == 0.0


def test_filter_metrics_f1_calculation():
    metrics = FilterMetrics(tp=4, fp=2, fn=2)
    assert metrics.precision == 2 / 3
    assert metrics.recall == 2 / 3
    assert metrics.f1_score == 2 / 3


def test_map_metrics_accuracy_zero_rows():
    metrics = MapMetrics(correct_by_column={"col": 0}, total_rows=0)
    assert metrics.accuracy == 0.0


def test_map_metrics_accuracy_empty_columns():
    metrics = MapMetrics(correct_by_column={}, total_rows=5)
    assert metrics.accuracy == 0.0
