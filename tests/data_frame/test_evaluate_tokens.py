import pytest

from evergreen.catalog.schema import Field, Schema
from evergreen.core.session_context import SessionContext
from evergreen.planner.logical.expr import col
from evergreen.provenance import Token
from evergreen.storage.row import Row


@pytest.fixture
def bool_schema() -> Schema:
    return Schema(
        (Field("id", int), Field("flag", bool), Field("value", int)),
        ("id",),
    )


@pytest.fixture
def bool_rows() -> list[Row]:
    return [
        Row((1, True, 10)),
        Row((2, False, 20)),
        Row((3, True, 30)),
    ]


def test_evaluate_tokens_all_valid(
    default_ctx: SessionContext, bool_rows: list[Row], bool_schema: Schema
):
    df = default_ctx.read_rows(bool_rows, bool_schema)

    tokens = [
        Token(row_id=(1,), predicate=col("flag"), sign=True),
        Token(row_id=(2,), predicate=col("flag"), sign=False),
        Token(row_id=(3,), predicate=col("flag"), sign=True),
    ]

    result = df.evaluate_tokens(tokens)

    assert all(result.values())
    assert df.count_valid_tokens(tokens) == 3


def test_evaluate_tokens_some_invalid(
    default_ctx: SessionContext, bool_rows: list[Row], bool_schema: Schema
):
    df = default_ctx.read_rows(bool_rows, bool_schema)

    tokens = [
        Token(row_id=(1,), predicate=col("flag"), sign=True),  # valid
        Token(row_id=(2,), predicate=col("flag"), sign=True),  # invalid - flag is False
        Token(row_id=(3,), predicate=col("flag"), sign=False),  # invalid - flag is True
    ]

    result = df.evaluate_tokens(tokens)

    assert result[tokens[0]] is True
    assert result[tokens[1]] is False
    assert result[tokens[2]] is False
    assert df.count_valid_tokens(tokens) == 1


def test_evaluate_tokens_missing_row(
    default_ctx: SessionContext, bool_rows: list[Row], bool_schema: Schema
):
    df = default_ctx.read_rows(bool_rows, bool_schema)

    tokens = [
        Token(row_id=(1,), predicate=col("flag"), sign=True),
        Token(row_id=(999,), predicate=col("flag"), sign=True),  # row doesn't exist
    ]

    result = df.evaluate_tokens(tokens)

    assert result[tokens[0]] is True
    assert result[tokens[1]] is False
    assert df.count_valid_tokens(tokens) == 1


def test_evaluate_tokens_with_comparison_predicate(
    default_ctx: SessionContext, bool_rows: list[Row], bool_schema: Schema
):
    df = default_ctx.read_rows(bool_rows, bool_schema)

    tokens = [
        Token(row_id=(1,), predicate=col("value") > 5, sign=True),  # 10 > 5 = True
        Token(
            row_id=(2,), predicate=col("value") > 25, sign=False
        ),  # 20 > 25 = False ✓
        Token(row_id=(3,), predicate=col("value") > 25, sign=True),  # 30 > 25 = True
    ]

    result = df.evaluate_tokens(tokens)

    assert all(result.values())
    assert df.count_valid_tokens(tokens) == 3


def test_evaluate_tokens_with_compound_predicate(
    default_ctx: SessionContext, bool_rows: list[Row], bool_schema: Schema
):
    df = default_ctx.read_rows(bool_rows, bool_schema)

    # flag=True AND value > 15
    predicate = col("flag") & (col("value") > 15)

    tokens = [
        Token(row_id=(1,), predicate=predicate, sign=False),  # True AND False = False
        Token(row_id=(2,), predicate=predicate, sign=False),  # False AND True = False
        Token(row_id=(3,), predicate=predicate, sign=True),  # True AND True = True
    ]

    result = df.evaluate_tokens(tokens)

    assert all(result.values())


def test_evaluate_tokens_with_equality_predicate(
    default_ctx: SessionContext, bool_rows: list[Row], bool_schema: Schema
):
    df = default_ctx.read_rows(bool_rows, bool_schema)

    tokens = [
        Token(row_id=(1,), predicate=col("value").eq(10), sign=True),
        Token(row_id=(2,), predicate=col("value").eq(10), sign=False),
        Token(row_id=(3,), predicate=col("value").ne(10), sign=True),
    ]

    result = df.evaluate_tokens(tokens)

    assert all(result.values())


def test_evaluate_tokens_empty(
    default_ctx: SessionContext, bool_rows: list[Row], bool_schema: Schema
):
    df = default_ctx.read_rows(bool_rows, bool_schema)

    result = df.evaluate_tokens([])

    assert result == {}
    assert df.count_valid_tokens([]) == 0
