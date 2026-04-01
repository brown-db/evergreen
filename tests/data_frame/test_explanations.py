import pytest

from evergreen.catalog.schema import Field, Schema
from evergreen.core.session_context import SessionContext
from evergreen.planner.logical.expr import bool_and, bool_or, col, count_if, proportion
from evergreen.provenance import Prov
from evergreen.storage.row import Row
from tests.helpers import validate_check_result


@pytest.mark.parametrize(
    "minimal_provenance,prov",
    [
        (
            False,
            Prov.pos_token((4,), col("is_true")) + Prov.pos_token((6,), col("is_true")),
        ),
        (True, Prov.pos_token((4,), col("is_true"))),
    ],
)
def test_bool_or_true(
    default_ctx: SessionContext, minimal_provenance: bool, prov: Prov
):
    if minimal_provenance:
        default_ctx.enable_minimal_provenance()

    df = default_ctx.read_json("tests/data/bool_or_true.jsonl", ("id",))
    result = (
        df.aggregate([bool_or(col("is_true")).alias("any")]).check(col("any")).collect()
    )

    validate_check_result(result, True, prov)


def test_bool_or_false(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/bool_or_false.jsonl", ("id",))
    result = (
        df.aggregate([bool_or(col("is_true")).alias("any")]).check(col("any")).collect()
    )
    prov = (
        Prov.neg_token((1,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
        * Prov.neg_token((4,), col("is_true"))
        * Prov.neg_token((5,), col("is_true"))
    )

    validate_check_result(result, False, prov)


def test_bool_and_true(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/bool_and_true.jsonl", ("id",))
    result = (
        df.aggregate([bool_and(col("is_true")).alias("all")])
        .check(col("all"))
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((2,), col("is_true"))
        * Prov.pos_token((3,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
    )

    validate_check_result(result, True, prov)


@pytest.mark.parametrize(
    "minimal_provenance,prov",
    [
        (
            False,
            Prov.neg_token((2,), col("is_true")) + Prov.neg_token((5,), col("is_true")),
        ),
        (True, Prov.neg_token((2,), col("is_true"))),
    ],
)
def test_bool_and_false(
    default_ctx: SessionContext, minimal_provenance: bool, prov: Prov
):
    if minimal_provenance:
        default_ctx.enable_minimal_provenance()

    df = default_ctx.read_json("tests/data/bool_and_false.jsonl", ("id",))
    result = (
        df.aggregate([bool_and(col("is_true")).alias("all")])
        .check(col("all"))
        .collect()
    )

    validate_check_result(result, False, prov)


def test_count_if_gt_true(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/count_if.jsonl", ("id",))
    result = (
        df.aggregate([count_if(col("is_true")).alias("count")])
        .check(col("count") > 1)
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true")) * Prov.pos_token((4,), col("is_true"))
    ) + (
        Prov.pos_token((1,), col("is_true")) * Prov.pos_token((5,), col("is_true"))
        + Prov.pos_token((4,), col("is_true")) * Prov.pos_token((5,), col("is_true"))
    )

    validate_check_result(result, True, prov)


def test_count_if_gt_false(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/count_if.jsonl", ("id",))
    result = (
        df.aggregate([count_if(col("is_true")).alias("count")])
        .check(col("count") > 3)
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
    )

    validate_check_result(result, False, prov)


def test_count_if_ge_true(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/count_if.jsonl", ("id",))
    result = (
        df.aggregate([count_if(col("is_true")).alias("count")])
        .check(col("count") >= 2)
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true")) * Prov.pos_token((4,), col("is_true"))
    ) + (
        Prov.pos_token((1,), col("is_true")) * Prov.pos_token((5,), col("is_true"))
        + Prov.pos_token((4,), col("is_true")) * Prov.pos_token((5,), col("is_true"))
    )

    validate_check_result(result, True, prov)


def test_count_if_ge_false(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/count_if.jsonl", ("id",))
    result = (
        df.aggregate([count_if(col("is_true")).alias("count")])
        .check(col("count") >= 4)
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
    )

    validate_check_result(result, False, prov)


def test_count_if_lt_true(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/count_if.jsonl", ("id",))
    result = (
        df.aggregate([count_if(col("is_true")).alias("count")])
        .check(col("count") < 4)
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
    )

    validate_check_result(result, True, prov)


def test_count_if_lt_false(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/count_if.jsonl", ("id",))
    result = (
        df.aggregate([count_if(col("is_true")).alias("count")])
        .check(col("count") < 2)
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true")) * Prov.pos_token((4,), col("is_true"))
    ) + (
        Prov.pos_token((1,), col("is_true")) * Prov.pos_token((5,), col("is_true"))
        + Prov.pos_token((4,), col("is_true")) * Prov.pos_token((5,), col("is_true"))
    )

    validate_check_result(result, False, prov)


def test_count_if_le_true(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/count_if.jsonl", ("id",))
    result = (
        df.aggregate([count_if(col("is_true")).alias("count")])
        .check(col("count") <= 3)
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
    )

    validate_check_result(result, True, prov)


def test_count_if_le_false(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/count_if.jsonl", ("id",))
    result = (
        df.aggregate([count_if(col("is_true")).alias("count")])
        .check(col("count") <= 1)
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true")) * Prov.pos_token((4,), col("is_true"))
    ) + (
        Prov.pos_token((1,), col("is_true")) * Prov.pos_token((5,), col("is_true"))
        + Prov.pos_token((4,), col("is_true")) * Prov.pos_token((5,), col("is_true"))
    )

    validate_check_result(result, False, prov)


def test_count_if_eq_true(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/count_if.jsonl", ("id",))
    result = (
        df.aggregate([count_if(col("is_true")).alias("count")])
        .check(col("count").eq(3))
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
    )

    validate_check_result(result, True, prov)


def test_count_if_eq_false(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/count_if.jsonl", ("id",))
    result = (
        df.aggregate([count_if(col("is_true")).alias("count")])
        .check(col("count").eq(2))
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
    )

    validate_check_result(result, False, prov)


def test_count_if_ne_true(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/count_if.jsonl", ("id",))
    result = (
        df.aggregate([count_if(col("is_true")).alias("count")])
        .check(col("count").ne(2))
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
    )

    validate_check_result(result, True, prov)


def test_count_if_ne_false(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/count_if.jsonl", ("id",))
    result = (
        df.aggregate([count_if(col("is_true")).alias("count")])
        .check(col("count").ne(3))
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
    )

    validate_check_result(result, False, prov)


def test_proportion_gt_true(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/proportion.jsonl", ("id",))
    result = (
        df.aggregate([proportion(col("is_true")).alias("proportion")])
        .check(col("proportion") > 0.2)
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true")) * Prov.pos_token((4,), col("is_true"))
    ) + (
        Prov.pos_token((1,), col("is_true")) * Prov.pos_token((5,), col("is_true"))
        + Prov.pos_token((4,), col("is_true")) * Prov.pos_token((5,), col("is_true"))
    )

    validate_check_result(result, True, prov)


def test_proportion_gt_false(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/proportion.jsonl", ("id",))
    result = (
        df.aggregate([proportion(col("is_true")).alias("proportion")])
        .check(col("proportion") > 0.6)
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
    )

    validate_check_result(result, False, prov)


def test_proportion_ge_true(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/proportion.jsonl", ("id",))
    result = (
        df.aggregate([proportion(col("is_true")).alias("proportion")])
        .check(col("proportion") >= 0.4)
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true")) * Prov.pos_token((4,), col("is_true"))
    ) + (
        Prov.pos_token((1,), col("is_true")) * Prov.pos_token((5,), col("is_true"))
        + Prov.pos_token((4,), col("is_true")) * Prov.pos_token((5,), col("is_true"))
    )

    validate_check_result(result, True, prov)


def test_proportion_ge_false(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/proportion.jsonl", ("id",))
    result = (
        df.aggregate([proportion(col("is_true")).alias("proportion")])
        .check(col("proportion") >= 0.8)
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
    )

    validate_check_result(result, False, prov)


def test_proportion_lt_true(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/proportion.jsonl", ("id",))
    result = (
        df.aggregate([proportion(col("is_true")).alias("proportion")])
        .check(col("proportion") < 0.8)
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
    )

    validate_check_result(result, True, prov)


def test_proportion_lt_false(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/proportion.jsonl", ("id",))
    result = (
        df.aggregate([proportion(col("is_true")).alias("proportion")])
        .check(col("proportion") < 0.4)
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true")) * Prov.pos_token((4,), col("is_true"))
    ) + (
        Prov.pos_token((1,), col("is_true")) * Prov.pos_token((5,), col("is_true"))
        + Prov.pos_token((4,), col("is_true")) * Prov.pos_token((5,), col("is_true"))
    )

    validate_check_result(result, False, prov)


def test_proportion_le_true(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/proportion.jsonl", ("id",))
    result = (
        df.aggregate([proportion(col("is_true")).alias("proportion")])
        .check(col("proportion") <= 0.6)
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
    )

    validate_check_result(result, True, prov)


def test_proportion_le_false(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/proportion.jsonl", ("id",))
    result = (
        df.aggregate([proportion(col("is_true")).alias("proportion")])
        .check(col("proportion") <= 0.2)
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true")) * Prov.pos_token((4,), col("is_true"))
    ) + (
        Prov.pos_token((1,), col("is_true")) * Prov.pos_token((5,), col("is_true"))
        + Prov.pos_token((4,), col("is_true")) * Prov.pos_token((5,), col("is_true"))
    )

    validate_check_result(result, False, prov)


def test_proportion_eq_true(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/proportion.jsonl", ("id",))
    result = (
        df.aggregate([proportion(col("is_true")).alias("proportion")])
        .check(col("proportion").eq(0.6))
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
    )

    validate_check_result(result, True, prov)


def test_proportion_eq_false(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/proportion.jsonl", ("id",))
    result = (
        df.aggregate([proportion(col("is_true")).alias("proportion")])
        .check(col("proportion").eq(0.4))
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
    )

    validate_check_result(result, False, prov)


def test_proportion_ne_true(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/proportion.jsonl", ("id",))
    result = (
        df.aggregate([proportion(col("is_true")).alias("proportion")])
        .check(col("proportion").ne(0.4))
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
    )

    validate_check_result(result, True, prov)


def test_proportion_ne_false(default_ctx: SessionContext):
    df = default_ctx.read_json("tests/data/proportion.jsonl", ("id",))
    result = (
        df.aggregate([proportion(col("is_true")).alias("proportion")])
        .check(col("proportion").ne(0.6))
        .collect()
    )
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((4,), col("is_true"))
        * Prov.pos_token((5,), col("is_true"))
        * Prov.neg_token((2,), col("is_true"))
        * Prov.neg_token((3,), col("is_true"))
    )

    validate_check_result(result, False, prov)


def test_empty_input_proportion():
    ctx = SessionContext()
    schema = Schema((Field("id", int), Field("flag", bool)), ("id",))
    df = ctx.read_rows([], schema)
    result = (
        df.aggregate([proportion(col("flag")).alias("prop")])
        .check(col("prop") >= 0.5)
        .collect()
    )
    validate_check_result(result, False, Prov.one())


def test_nested_quantifiers(default_ctx: SessionContext):
    schema = Schema(
        (Field("id", int), Field("group", str), Field("is_true", bool)), ("id",)
    )
    rows = [
        Row((1, "A", True)),
        Row((2, "B", False)),
        Row((3, "A", False)),
        Row((4, "C", True)),
        Row((5, "A", True)),
        Row((6, "B", True)),
        Row((7, "B", True)),
        Row((8, "C", False)),
        Row((9, "A", True)),
        Row((10, "B", False)),
    ]
    df = default_ctx.read_rows(rows, schema)
    result = (
        df.aggregate(
            [count_if(col("is_true")).alias("num_true")], group_by=[col("group")]
        )
        .aggregate([bool_or(col("num_true") > 1).alias("exists")])
        .check(col("exists"))
        .collect()
    )
    prov = (
        (Prov.pos_token((1,), col("is_true")) * Prov.pos_token((5,), col("is_true")))
        + (Prov.pos_token((1,), col("is_true")) * Prov.pos_token((9,), col("is_true")))
        + (Prov.pos_token((5,), col("is_true")) * Prov.pos_token((9,), col("is_true")))
    ) + (Prov.pos_token((6,), col("is_true")) * Prov.pos_token((7,), col("is_true")))

    validate_check_result(result, True, prov)


def test_with_rank_check_rank_1(default_ctx: SessionContext):
    """Test provenance for rank 1 with 3 groups."""
    schema = Schema(
        (Field("id", int), Field("group", str), Field("is_true", bool)), ("id",)
    )
    rows = [
        Row((1, "A", True)),
        Row((2, "A", True)),  # A: count=2 (rank 1)
        Row((3, "B", True)),
        Row((4, "B", False)),  # B: count=1 (rank 2)
        Row((5, "C", False)),  # C: count=0 (rank 3)
    ]
    df = default_ctx.read_rows(rows, schema)
    result = (
        df.aggregate(
            [count_if(col("is_true")).alias("num_true")], group_by=[col("group")]
        )
        .with_rank(col("num_true"))
        .filter(col("group").eq("A"))
        .check(col("rank").eq(1))
        .collect()
    )
    # A's rank is justified by:
    #   - A > B (2 > 1): pos(1) * pos(2) * pos(3) * neg(4)
    #   - A > C (2 > 0): (pos(1) + pos(2)) * neg(5)
    # After DNF absorption: pos(1) * pos(2) * pos(3) * neg(4) * neg(5)
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((2,), col("is_true"))
        * Prov.pos_token((3,), col("is_true"))
        * Prov.neg_token((4,), col("is_true"))
        * Prov.neg_token((5,), col("is_true"))
    )

    validate_check_result(result, True, prov, row_id="A")


def test_with_rank_check_rank_2(default_ctx: SessionContext):
    """Test provenance for rank 2 with 3 groups."""
    schema = Schema(
        (Field("id", int), Field("group", str), Field("is_true", bool)), ("id",)
    )
    rows = [
        Row((1, "A", True)),
        Row((2, "A", True)),  # A: count=2 (rank 1)
        Row((3, "B", True)),
        Row((4, "B", False)),  # B: count=1 (rank 2)
        Row((5, "C", False)),  # C: count=0 (rank 3)
    ]
    df = default_ctx.read_rows(rows, schema)
    result = (
        df.aggregate(
            [count_if(col("is_true")).alias("num_true")], group_by=[col("group")]
        )
        .with_rank(col("num_true"))
        .filter(col("group").eq("B"))
        .check(col("rank").eq(2))
        .collect()
    )
    # B's rank is justified by:
    #   - B < A (1 < 2): pos(1) * pos(2) * pos(3) * neg(4)
    #   - B > C (1 > 0): pos(3) * neg(5)
    # After DNF absorption: pos(1) * pos(2) * pos(3) * neg(4) * neg(5)
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((2,), col("is_true"))
        * Prov.pos_token((3,), col("is_true"))
        * Prov.neg_token((4,), col("is_true"))
        * Prov.neg_token((5,), col("is_true"))
    )

    validate_check_result(result, True, prov, row_id="B")


def test_with_rank_check_rank_3(default_ctx: SessionContext):
    """Test provenance for rank 3 - different from rank 1 and 2!"""
    schema = Schema(
        (Field("id", int), Field("group", str), Field("is_true", bool)), ("id",)
    )
    rows = [
        Row((1, "A", True)),
        Row((2, "A", True)),  # A: count=2 (rank 1)
        Row((3, "B", True)),
        Row((4, "B", False)),  # B: count=1 (rank 2)
        Row((5, "C", False)),  # C: count=0 (rank 3)
    ]
    df = default_ctx.read_rows(rows, schema)
    result = (
        df.aggregate(
            [count_if(col("is_true")).alias("num_true")], group_by=[col("group")]
        )
        .with_rank(col("num_true"))
        .filter(col("group").eq("C"))
        .check(col("rank").eq(3))
        .collect()
    )
    # C's rank is justified by:
    #   - C < A (0 < 2): (pos(1) + pos(2)) * neg(5)
    #   - C < B (0 < 1): pos(3) * neg(5)
    # Combined: (pos(1) * pos(3) * neg(5)) + (pos(2) * pos(3) * neg(5))
    # Note: This is DIFFERENT from A and B's provenance!
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((3,), col("is_true"))
        * Prov.neg_token((5,), col("is_true"))
    ) + (
        Prov.pos_token((2,), col("is_true"))
        * Prov.pos_token((3,), col("is_true"))
        * Prov.neg_token((5,), col("is_true"))
    )

    validate_check_result(result, True, prov, row_id="C")


def test_with_rank_check_false(default_ctx: SessionContext):
    """Test provenance when the rank check fails."""
    schema = Schema(
        (Field("id", int), Field("group", str), Field("is_true", bool)), ("id",)
    )
    rows = [
        Row((1, "A", True)),
        Row((2, "A", True)),  # A: count=2 (rank 1)
        Row((3, "B", True)),
        Row((4, "B", False)),  # B: count=1 (rank 2)
        Row((5, "C", False)),  # C: count=0 (rank 3)
    ]
    df = default_ctx.read_rows(rows, schema)
    result = (
        df.aggregate(
            [count_if(col("is_true")).alias("num_true")], group_by=[col("group")]
        )
        .with_rank(col("num_true"))
        .filter(col("group").eq("C"))
        .check(col("rank").eq(1))  # C is not rank 1!
        .collect()
    )
    # Provenance still explains C's actual rank (3)
    prov = (
        Prov.pos_token((1,), col("is_true"))
        * Prov.pos_token((3,), col("is_true"))
        * Prov.neg_token((5,), col("is_true"))
    ) + (
        Prov.pos_token((2,), col("is_true"))
        * Prov.pos_token((3,), col("is_true"))
        * Prov.neg_token((5,), col("is_true"))
    )

    validate_check_result(result, False, prov, row_id="C")
