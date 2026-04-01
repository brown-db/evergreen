import logging
import random
import re

import pytest
from pytest import LogCaptureFixture

from evergreen.catalog.schema import Field, Schema
from evergreen.core.session_context import SessionContext
from evergreen.data_frame import QueryResult
from evergreen.planner.logical.expr import (
    Expr,
    bool_and,
    bool_or,
    col,
    count_if,
    prompt,
    proportion,
)
from evergreen.storage.row import Row
from tests.helpers import add_embeddings


@pytest.fixture
def schema() -> Schema:
    return Schema((Field("id", int), Field("flag", bool)), ("id",))


def validate_check_result(result: QueryResult, is_valid: bool, max_token_count: int):
    rows = result.rows
    assert len(rows) == 1
    assert rows[0][1] is is_valid
    monomials = rows[0].get_prov_monomials(1)
    assert len(monomials) == 1
    assert len(monomials[0]) <= max_token_count


class TestEstimation:
    @pytest.mark.parametrize(
        "num_true,is_valid,max_token_count",
        [
            # True cases: at least one True exists (deterministic early stop)
            (50, True, 1),
            (10, True, 1),
            (1, True, 1),
            # False cases: no True values (estimation concludes False)
            (0, False, 100),  # Need to scan more to conclude False
        ],
    )
    def test_existential_quantifier(
        self,
        ctx_with_estimation: SessionContext,
        schema: Schema,
        num_true: int,
        is_valid: bool,
        max_token_count: int,
    ):
        rows = [Row((i, i < num_true)) for i in range(100)]
        random.shuffle(rows)
        df = ctx_with_estimation.read_rows(rows, schema)
        result = (
            df.aggregate([bool_or(col("flag")).alias("any_true")])
            .check(col("any_true"))
            .collect()
        )

        validate_check_result(result, is_valid, max_token_count)

    @pytest.mark.parametrize(
        "true_proportion,threshold,is_valid,max_token_count",
        [
            # Max token counts are based on empirical observations
            # True cases: proportion >= threshold
            (0.5, 0.5, True, 50),
            (0.6, 0.5, True, 50),
            (0.7, 0.5, True, 49),
            (0.8, 0.5, True, 35),
            # False cases: proportion < threshold
            (0.3, 0.5, False, 80),
            (0.4, 0.5, False, 95),
            (0.45, 0.5, False, 100),
            (0.49, 0.5, False, 100),
        ],
    )
    def test_proportional_quantifier(
        self,
        ctx_with_estimation: SessionContext,
        schema: Schema,
        true_proportion: float,
        is_valid: bool,
        threshold: float,
        max_token_count: int,
    ):
        rows = [Row((i, i < true_proportion * 100)) for i in range(100)]
        random.shuffle(rows)
        df = ctx_with_estimation.read_rows(rows, schema)
        result = (
            df.aggregate([proportion(col("flag")).alias("prop")])
            .check(col("prop") >= threshold)
            .collect()
        )

        validate_check_result(result, is_valid, max_token_count)

    @pytest.mark.parametrize(
        "true_count,threshold,is_valid,max_token_count",
        [
            # Max token counts are based on empirical observations
            # True cases: count <= threshold
            (20, 50, True, 50),
            (30, 50, True, 80),
            (40, 50, True, 90),
            (50, 50, True, 100),
            # False cases: count > threshold
            (51, 50, False, 51),
            (60, 50, False, 51),
            (70, 50, False, 48),
            (80, 50, False, 30),
        ],
    )
    def test_cardinal_quantifier(
        self,
        ctx_with_estimation: SessionContext,
        schema: Schema,
        true_count: int,
        threshold: int,
        is_valid: bool,
        max_token_count: int,
    ):
        rows = [Row((i, i < true_count)) for i in range(100)]
        random.shuffle(rows)
        df = ctx_with_estimation.read_rows(rows, schema)
        result = (
            df.aggregate([count_if(col("flag")).alias("cnt")])
            .check(col("cnt") <= threshold)
            .collect()
        )

        validate_check_result(result, is_valid, max_token_count)

    @pytest.mark.parametrize(
        "num_true,is_valid,max_token_count",
        [
            # True cases: all values are True
            # Max token count of 75 is based on empirical observations
            (100, True, 75),  # All True
            # False cases: some values are False (early stop on first False)
            (90, False, 1),
            (50, False, 1),
            (10, False, 1),
        ],
    )
    def test_universal_quantifier(
        self,
        ctx_with_estimation: SessionContext,
        schema: Schema,
        num_true: int,
        is_valid: bool,
        max_token_count: int,
    ):
        rows = [Row((i, i < num_true)) for i in range(100)]
        random.shuffle(rows)
        df = ctx_with_estimation.read_rows(rows, schema)
        result = (
            df.aggregate([bool_and(col("flag")).alias("all_true")])
            .check(col("all_true"))
            .collect()
        )

        validate_check_result(result, is_valid, max_token_count)


class TestConfidenceLevelAllocation:
    @pytest.mark.parametrize(
        "agg_expr,check_predicate",
        [
            (bool_or(col("flag")).alias("some"), col("some")),
            (proportion(col("flag")).alias("prop"), col("prop") > 0.3),
            (count_if(col("flag")).alias("cnt"), col("cnt") > 30),
            (bool_and(col("flag")).alias("all"), col("all")),
        ],
    )
    def test_single_agg_expr_supporting_estimation(
        self,
        ctx_with_estimation: SessionContext,
        schema: Schema,
        caplog: LogCaptureFixture,
        agg_expr: Expr,
        check_predicate: Expr,
    ):
        rows = [Row((i, i < 50)) for i in range(100)]
        random.shuffle(rows)
        df = ctx_with_estimation.read_rows(rows, schema)
        with caplog.at_level(logging.DEBUG):
            df.aggregate([agg_expr]).check(check_predicate).collect()

        results = self._parse_confidence_levels(caplog)
        assert len(results) == 1

        confidence_level, accumulator_confidence_level = results[0]
        assert confidence_level == 0.95
        assert accumulator_confidence_level == 0.95

    @pytest.mark.parametrize(
        "agg_exprs,check_predicate,accumulator_confidence_level",
        [
            (
                [
                    proportion(col("flag")).alias("prop"),
                    count_if(col("flag")).alias("cnt"),
                ],
                (col("prop") > 0.3) | (col("cnt") > 50),
                0.975,
            ),
            (
                [
                    proportion(col("flag")).alias("prop"),
                    bool_or(col("flag")).alias("any"),
                ],
                (col("prop") > 0.3) & (col("any")),
                0.975,
            ),
        ],
    )
    def test_multiple_agg_exprs(
        self,
        ctx_with_estimation: SessionContext,
        schema: Schema,
        caplog: LogCaptureFixture,
        agg_exprs: list[Expr],
        check_predicate: Expr,
        accumulator_confidence_level: float,
    ):
        # We need to enable minimal provenance here because the check predicate
        # may contain comparisons that cause accumulators to stop at different
        # points; thus, their provenance may not be minimal (possibly causing a
        # combinatorial explosion).
        ctx_with_estimation.enable_minimal_provenance()

        rows = [Row((i, i < 50)) for i in range(100)]
        random.shuffle(rows)
        df = ctx_with_estimation.read_rows(rows, schema)
        with caplog.at_level(logging.DEBUG):
            df.aggregate(agg_exprs).check(check_predicate).collect()

        results = self._parse_confidence_levels(caplog)
        assert len(results) == 1

        result_confidence_level, result_accumulator_confidence_level = results[0]
        assert result_confidence_level == 0.95
        assert result_accumulator_confidence_level == accumulator_confidence_level

    def test_group_by_aggregate_supporting_estimation(
        self,
        ctx_with_estimation: SessionContext,
        caplog: LogCaptureFixture,
    ):
        schema = Schema(
            (Field("id", int), Field("group", str), Field("flag", bool)), ("id",)
        )
        rows = [
            Row((1, "A", True)),
            Row((2, "A", False)),
            Row((3, "B", True)),
            Row((4, "B", True)),
        ]
        df = ctx_with_estimation.read_rows(rows, schema)
        with caplog.at_level(logging.DEBUG):
            (
                df.aggregate(
                    [proportion(col("flag")).alias("prop")],
                    group_by=[col("group")],
                )
                .aggregate([bool_and(col("prop") >= 0.5).alias("all_above_half")])
                .check(col("all_above_half"))
                .collect()
            )

        results = self._parse_confidence_levels(caplog)
        assert len(results) == 3

        # We check the confidence levels in the following order due to the fact
        # that bool_and creates its accumulator after the proportion accumulator
        # for the first group emits. The proportion accumulator for the second
        # group is created after the bool_and accumulator.

        # Conficence levels for proportion of first group
        confidence_level_0, accumulator_confidence_level_0 = results[0]
        assert confidence_level_0 == 0.975
        assert accumulator_confidence_level_0 == 0.9875

        # Confidence levels for bool_and
        confidence_level_1, accumulator_confidence_level_1 = results[1]
        assert confidence_level_1 == 0.975
        assert accumulator_confidence_level_1 == 0.975

        # Confidence levels for proportion of second group
        confidence_level_2, accumulator_confidence_level_2 = results[2]
        assert confidence_level_2 == 0.975
        assert accumulator_confidence_level_2 == 0.9875

    def test_group_by_aggregate_with_bool_or(
        self,
        ctx_with_estimation: SessionContext,
        caplog: LogCaptureFixture,
    ):
        schema = Schema(
            (Field("id", int), Field("group", str), Field("flag", bool)), ("id",)
        )
        rows = [Row((1, "A", True)), Row((2, "A", False)), Row((3, "B", True))]
        df = ctx_with_estimation.read_rows(rows, schema)
        with caplog.at_level(logging.DEBUG):
            (
                df.aggregate(
                    [proportion(col("flag")).alias("prop")],
                    group_by=[col("group")],
                )
                .aggregate([bool_or(col("prop") >= 0.5).alias("any_above_half")])
                .check(col("any_above_half"))
                .collect()
            )
        results = self._parse_confidence_levels(caplog)
        assert len(results) == 2

        # Confidence levels for proportion of group the first group
        confidence_level_0, accumulator_confidence_level_0 = results[0]
        assert confidence_level_0 == 0.975
        assert accumulator_confidence_level_0 == 0.9875

        # Confidence levels for bool_or
        confidence_level_1, accumulator_confidence_level_1 = results[1]
        assert confidence_level_1 == 0.975
        assert accumulator_confidence_level_1 == 0.975

    def test_nested_aggregate_with_estimation_and_relevance_sort(
        self,
        ctx_with_relevance_sort: SessionContext,
        caplog: LogCaptureFixture,
    ):
        ctx_with_relevance_sort.enable_estimation()

        schema = Schema(
            (Field("id", int), Field("group", str), Field("review", str)), ("id",)
        )
        rows = [
            Row((1, "A", "The burgers here are great.")),
            Row((2, "A", "I love the fries.")),
        ]
        rows, schema = add_embeddings("review", rows, schema)
        df = ctx_with_relevance_sort.read_rows(rows, schema)

        with caplog.at_level(logging.DEBUG):
            (
                df.map(
                    prompt("identify whether the {review} mentions burgers").alias(
                        "mentions_burgers"
                    )
                )
                .aggregate(
                    [count_if(col("mentions_burgers")).alias("burger_count")],
                    group_by=[col("group")],
                )
                .aggregate(
                    [bool_and(col("burger_count") > 0).alias("all_groups_have_burgers")]
                )
                .check(col("all_groups_have_burgers"))
                .collect()
            )

        # Only outer aggregate (BoolAnd with Shuffle) gets confidence level
        # Inner aggregate (CountIf with RelevanceSort) does NOT support estimation
        results = self._parse_confidence_levels(caplog)
        assert len(results) == 1

        confidence_level, accumulator_confidence_level = results[0]
        assert confidence_level == 0.95
        assert accumulator_confidence_level == 0.95

    def test_aggregate_with_fusion(
        self,
        ctx_with_fusion: SessionContext,
        caplog: LogCaptureFixture,
    ):
        ctx_with_fusion.enable_estimation()

        schema = Schema((Field("id", int), Field("review", str)), ("id",))
        rows = [
            Row((1, "The burgers at this place are amazing.")),
            Row((2, "The burgers at this place are okay. Not the best in town.")),
            Row((3, "The burgers at this place are terrible. The fries are soggy.")),
            Row((4, "The pizza is lovely here.")),
        ]
        df = ctx_with_fusion.read_rows(rows, schema)

        with caplog.at_level(logging.DEBUG):
            (
                df.filter(prompt("the {review} mentions burgers"))
                .map(
                    prompt("identify whether the {review} enjoys the burgers").alias(
                        "enjoys_burgers"
                    )
                )
                .aggregate([proportion(col("enjoys_burgers")).alias("prop")])
                .check(col("prop") > 0.5)
                .collect()
            )

        results = self._parse_confidence_levels(caplog)
        assert len(results) == 1

        confidence_level, accumulator_confidence_level = results[0]
        assert confidence_level == 0.95
        assert accumulator_confidence_level == 0.95

    def test_aggregate_with_relevance_sort(
        self,
        ctx_with_relevance_sort: SessionContext,
        caplog: LogCaptureFixture,
    ):
        ctx_with_relevance_sort.enable_estimation()

        schema = Schema((Field("id", int), Field("review", str)), ("id",))
        rows = [
            Row((1, "The burgers at this place are amazing.")),
        ]
        rows, schema = add_embeddings("review", rows, schema)
        df = ctx_with_relevance_sort.read_rows(rows, schema)
        with caplog.at_level(logging.DEBUG):
            (
                df.map(
                    prompt("identify whether the {review} mentions burgers").alias(
                        "mentions_burgers"
                    )
                )
                .aggregate(
                    [bool_and(col("mentions_burgers")).alias("all_mention_burgers")]
                )
                .check(col("all_mention_burgers"))
                .collect()
            )

        results = self._parse_confidence_levels(caplog)
        assert len(results) == 1

        confidence_level, accumulator_confidence_level = results[0]
        assert confidence_level == 0.95
        assert accumulator_confidence_level == 0.95

    @staticmethod
    def _parse_confidence_levels(
        caplog: LogCaptureFixture,
    ) -> list[tuple[float, float]]:
        results: list[tuple[float, float]] = []

        for record in caplog.records:
            if "aggregate confidence level allocation" in record.message:
                msg = record.message
                confidence_level_match = re.search(r"confidence_level=([\d.]+)", msg)
                accumulator_confidence_level_match = re.search(
                    r"accumulator_confidence_level=([\d.]+)", msg
                )

                if (
                    confidence_level_match is not None
                    and accumulator_confidence_level_match is not None
                ):
                    results.append(
                        (
                            float(confidence_level_match.group(1)),
                            float(accumulator_confidence_level_match.group(1)),
                        )
                    )

        return results
