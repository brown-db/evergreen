import pytest

from evergreen.catalog.schema import Field, Schema
from evergreen.core.session_context import SessionContext
from evergreen.planner.logical.expr import (
    bool_and,
    bool_or,
    col,
    count_if,
    prompt,
    proportion,
)
from evergreen.planner.logical.plan import (
    Aggregate,
    Check,
    CountScan,
    Filter,
    FusedFilterProjection,
    Log,
    Projection,
    RelevanceSort,
    Shuffle,
    Sort,
    TableScan,
)
from tests.helpers import LogicalPlanPattern, P


@pytest.fixture
def schema() -> Schema:
    return Schema(
        (Field("id", str), Field("business_id", str), Field("text", str)), ("id",)
    )


@pytest.mark.parametrize(
    "optimized,plan_pattern",
    [
        (False, P(Check, P(Aggregate, P(Log, P(Projection, P(TableScan)))))),
        (
            True,
            P(
                Check,
                P(
                    Aggregate,
                    P(Log, P(Projection, P(RelevanceSort, P(CountScan, P(TableScan))))),
                ),
            ),
        ),
    ],
)
def test_existential_claim(
    optimized_ctx: SessionContext,
    schema: Schema,
    optimized: bool,
    plan_pattern: LogicalPlanPattern,
):
    ctx = SessionContext()

    if optimized:
        ctx = optimized_ctx

    df = ctx.read_rows([], schema)
    df = (
        df.map(
            prompt(
                "identify whether the review {text} mentions waiting 20+ minutes to be "
                "acknowledged",
                bool,
            ).alias("is_long_wait")
        )
        .log("")
        .aggregate([bool_or(col("is_long_wait")).alias("exists_long_wait")])
        .check(col("exists_long_wait"))
    )

    plan_pattern.validate_match(df.optimized_logical_plan())


@pytest.mark.parametrize(
    "optimized,plan_pattern",
    [
        (False, P(Check, P(Aggregate, P(Log, P(Projection, P(TableScan)))))),
        (
            True,
            P(
                Check,
                P(
                    Aggregate,
                    P(Log, P(Projection, P(RelevanceSort, P(CountScan, P(TableScan))))),
                ),
            ),
        ),
    ],
)
def test_cardinal_claim(
    optimized_ctx: SessionContext,
    schema: Schema,
    optimized: bool,
    plan_pattern: LogicalPlanPattern,
):
    ctx = SessionContext()

    if optimized:
        ctx = optimized_ctx

    df = ctx.read_rows([], schema)
    df = (
        df.map(
            prompt("identify whether the {text} mentions sewer smell problems").alias(
                "has_sewer_problems"
            )
        )
        .log("")
        .aggregate([count_if(col("has_sewer_problems")).alias("num_sewer_problems")])
        .check(col("num_sewer_problems") >= 2)
    )

    plan_pattern.validate_match(df.optimized_logical_plan())


@pytest.mark.parametrize(
    "optimized,plan_pattern",
    [
        (
            False,
            P(
                Check,
                P(Aggregate, P(Log, P(Projection, P(Filter, P(Log, P(TableScan)))))),
            ),
        ),
        (
            True,
            P(
                Check,
                P(
                    Aggregate,
                    P(
                        Log,
                        P(
                            FusedFilterProjection,
                            P(Filter, P(Log, P(RelevanceSort, P(TableScan)))),
                        ),
                    ),
                ),
            ),
        ),
    ],
)
def test_cardinal_claim_with_filter(
    optimized_ctx: SessionContext,
    schema: Schema,
    optimized: bool,
    plan_pattern: LogicalPlanPattern,
):
    ctx = SessionContext()

    if optimized:
        ctx = optimized_ctx

    df = ctx.read_rows([], schema)
    df = (
        df.log("")
        .filter(prompt("the review {text} mentions the restaurant's fish and chips"))
        .map(
            prompt(
                "identify whether the review {text} calls the fish and chips the best "
                "in town",
                bool,
            ).alias("is_best")
        )
        .log("")
        .aggregate([count_if(col("is_best")).alias("best_count")])
        .check(col("best_count") >= 30)
    )

    plan_pattern.validate_match(df.optimized_logical_plan())


@pytest.mark.parametrize(
    "optimized,plan_pattern",
    [
        (False, P(Check, P(Aggregate, P(Log, P(Projection, P(TableScan)))))),
        (
            True,
            P(
                Check,
                P(
                    Aggregate,
                    P(Log, P(Projection, P(Shuffle, P(CountScan, P(TableScan))))),
                ),
            ),
        ),
    ],
)
def test_proportional_claim(
    optimized_ctx: SessionContext,
    schema: Schema,
    optimized: bool,
    plan_pattern: LogicalPlanPattern,
):
    ctx = SessionContext()

    if optimized:
        ctx = optimized_ctx

    df = ctx.read_rows([], schema)
    df = (
        df.map(
            prompt(
                "identify whether the review {text} has positive sentiment",
                bool,
            ).alias("is_positive")
        )
        .log("")
        .aggregate([proportion(col("is_positive")).alias("positive_prop")])
        .check(col("positive_prop") > 0.5)
    )

    plan_pattern.validate_match(df.optimized_logical_plan())


@pytest.mark.parametrize(
    "optimized,plan_pattern",
    [
        (
            False,
            P(
                Check,
                P(Aggregate, P(Log, P(Projection, P(Filter, P(Log, P(TableScan)))))),
            ),
        ),
        (
            True,
            P(
                Check,
                P(
                    Aggregate,
                    P(
                        Log,
                        P(
                            FusedFilterProjection,
                            P(Filter, P(Log, P(Shuffle, P(TableScan)))),
                        ),
                    ),
                ),
            ),
        ),
    ],
)
def test_proportional_claim_with_filter(
    optimized_ctx: SessionContext,
    schema: Schema,
    optimized: bool,
    plan_pattern: LogicalPlanPattern,
):
    ctx = SessionContext()

    if optimized:
        ctx = optimized_ctx

    df = ctx.read_rows([], schema)
    df = (
        df.log("")
        .filter(prompt("the review {text} mentions the restaurant's fish and chips"))
        .map(
            prompt(
                "identify whether the review {text} calls the fish and chips the best "
                "in town",
                bool,
            ).alias("is_best")
        )
        .log("")
        .aggregate([proportion(col("is_best")).alias("best_prop")])
        .check(col("best_prop") >= 0.1)
    )

    plan_pattern.validate_match(df.optimized_logical_plan())


@pytest.mark.parametrize(
    "optimized,plan_pattern",
    [
        (
            False,
            P(
                Check,
                P(Aggregate, P(Log, P(Projection, P(Filter, P(Log, P(TableScan)))))),
            ),
        ),
        (
            True,
            P(
                Check,
                P(
                    Aggregate,
                    P(
                        Log,
                        P(
                            FusedFilterProjection,
                            P(Filter, P(Log, P(Shuffle, P(TableScan)))),
                        ),
                    ),
                ),
            ),
        ),
    ],
)
def test_universal_claim(
    optimized_ctx: SessionContext,
    schema: Schema,
    optimized: bool,
    plan_pattern: LogicalPlanPattern,
):
    ctx = SessionContext()

    if optimized:
        ctx = optimized_ctx

    df = ctx.read_rows([], schema)
    df = (
        df.log("")
        .filter(
            prompt("the review {text} mentions the price for happy hour appetizers")
        )
        .map(
            prompt(
                "identify whether the review {text} says that the price of happy hour "
                "appetizers is $6",
                bool,
            ).alias("is_six_dollars")
        )
        .log("")
        .aggregate([bool_and(col("is_six_dollars")).alias("all_six_dollars")])
        .check(col("all_six_dollars"))
    )

    plan_pattern.validate_match(df.optimized_logical_plan())


@pytest.mark.parametrize(
    "optimized,plan_pattern",
    [
        (
            False,
            P(
                Check,
                P(
                    Aggregate,
                    P(Aggregate, P(Log, P(Projection, P(Sort, P(TableScan))))),
                ),
            ),
        ),
        (
            True,
            P(
                Check,
                P(
                    Aggregate,
                    P(
                        Aggregate,
                        P(
                            Log,
                            P(
                                Projection,
                                P(Shuffle, P(Sort, P(RelevanceSort, P(TableScan)))),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ],
)
def test_nested_quantification_with_map(
    optimized_ctx: SessionContext,
    schema: Schema,
    optimized: bool,
    plan_pattern: LogicalPlanPattern,
):
    ctx = SessionContext()

    if optimized:
        ctx = optimized_ctx

    df = ctx.read_rows([], schema)
    df = (
        df.map(
            prompt(
                "The {text} mentions inconsistent order fulfillment",
                bool,
            ).alias("mentions_inconsistent_order")
        )
        .log("")
        .aggregate(
            [bool_or(col("mentions_inconsistent_order")).alias("location_has_issue")],
            group_by=[col("business_id")],
        )
        .aggregate(
            [proportion(col("location_has_issue")).alias("prop_locations_with_issue")]
        )
        .check(col("prop_locations_with_issue") > 0.5)
    )

    plan_pattern.validate_match(df.optimized_logical_plan())


@pytest.mark.parametrize(
    "optimized,plan_pattern",
    [
        (
            False,
            P(
                Check,
                P(
                    Aggregate,
                    P(
                        Aggregate,
                        P(Log, P(Projection, P(Sort, P(Filter, P(Log, P(TableScan)))))),
                    ),
                ),
            ),
        ),
        (
            True,
            P(
                Check,
                P(
                    Aggregate,
                    P(
                        Aggregate,
                        P(
                            Log,
                            P(
                                FusedFilterProjection,
                                P(Filter, P(Log, P(Shuffle, P(Sort, P(TableScan))))),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ],
)
def test_nested_quantification_with_map_and_filter(
    optimized_ctx: SessionContext,
    schema: Schema,
    optimized: bool,
    plan_pattern: LogicalPlanPattern,
):
    ctx = SessionContext()

    if optimized:
        ctx = optimized_ctx

    df = ctx.read_rows([], schema)
    df = (
        df.log("")
        .filter(prompt("The {text} mentions interactions with staff or employees"))
        .map(
            prompt(
                "Identify whether the {text} describes the staff as unhelpful, "
                "rude, or providing poor service",
                bool,
            ).alias("staff_unhelpful")
        )
        .log("")
        .aggregate(
            [proportion(col("staff_unhelpful")).alias("unhelpful_prop")],
            group_by=[col("business_id")],
        )
        .aggregate(
            [bool_or(col("unhelpful_prop") >= 0.5).alias("some_consistently_unhelpful")]
        )
        .check(col("some_consistently_unhelpful"))
    )

    plan_pattern.validate_match(df.optimized_logical_plan())


@pytest.mark.parametrize(
    "optimized,plan_pattern",
    [
        (
            False,
            P(
                Check,
                P(
                    Aggregate,
                    P(Aggregate, P(Log, P(Projection, P(Sort, P(TableScan))))),
                ),
            ),
        ),
        (
            True,
            P(
                Check,
                P(
                    Aggregate,
                    P(
                        Aggregate,
                        P(
                            Log,
                            P(
                                Projection,
                                P(Shuffle, P(Sort, P(RelevanceSort, P(TableScan)))),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ],
)
def test_nested_with_bool_or(
    optimized_ctx: SessionContext,
    schema: Schema,
    optimized: bool,
    plan_pattern: LogicalPlanPattern,
):
    ctx = SessionContext()

    if optimized:
        ctx = optimized_ctx

    df = ctx.read_rows([], schema)

    df = (
        df.map(
            prompt(
                "Identify whether the {text} mentions waiting 20+ minutes",
                bool,
            ).alias("mentions_long_wait")
        )
        .log("")
        .aggregate(
            [bool_or(col("mentions_long_wait")).alias("location_has_long_wait")],
            group_by=[col("business_id")],
        )
        .aggregate(
            [
                # Outer bool_or uses estimation
                bool_or(col("location_has_long_wait")).alias(
                    "some_location_has_long_wait"
                )
            ]
        )
        .check(col("some_location_has_long_wait"))
    )

    plan_pattern.validate_match(df.optimized_logical_plan())


@pytest.mark.parametrize(
    "optimized,plan_pattern",
    [
        (
            False,
            P(
                Check,
                P(
                    Aggregate,
                    P(Aggregate, P(Log, P(Projection, P(Sort, P(TableScan))))),
                ),
            ),
        ),
        (
            True,
            P(
                Check,
                P(
                    Aggregate,
                    P(
                        Aggregate,
                        P(
                            Log,
                            P(
                                Projection,
                                P(Shuffle, P(Sort, P(RelevanceSort, P(TableScan)))),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ],
)
def test_nested_quantification_with_bool_and(
    optimized_ctx: SessionContext,
    schema: Schema,
    optimized: bool,
    plan_pattern: LogicalPlanPattern,
):
    ctx = SessionContext()

    if optimized:
        ctx = optimized_ctx

    df = ctx.read_rows([], schema)

    df = (
        df.map(
            prompt(
                "Identify whether the {text} reports or mentions long wait times",
                bool,
            ).alias("mentions_long_wait")
        )
        .log("")
        .aggregate(
            [bool_or(col("mentions_long_wait")).alias("has_long_wait_report")],
            group_by=[col("business_id")],
        )
        .aggregate(
            [bool_and(col("has_long_wait_report")).alias("all_locations_have_reports")]
        )
        .check(col("all_locations_have_reports"))
    )

    plan_pattern.validate_match(df.optimized_logical_plan())
