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
from evergreen.provenance import Prov
from evergreen.storage.row import Row
from tests.helpers import validate_check_result


@pytest.mark.parametrize(
    "early_stop_enabled,prov",
    [
        pytest.param(
            False,
            Prov.pos_token((3,), col("is_long_wait"))
            + Prov.pos_token((5,), col("is_long_wait")),
            id="without_early_stop",
        ),
        pytest.param(
            True,
            Prov.pos_token((3,), col("is_long_wait")),
            id="with_early_stop",
        ),
    ],
)
def test_existential_claim(
    ctx_with_model: SessionContext, early_stop_enabled: bool, prov: Prov
):
    if early_stop_enabled:
        ctx_with_model.enable_early_stop()

    schema = Schema((Field("id", int), Field("review", str)), ("id",))
    rows = [
        Row((1, "The ice cream was delicious!")),
        Row((2, "The service was fast. I only waited for 1 minute.")),
        Row((3, "I had to wait so long (~20 min) to get the server's attention.")),
        Row((4, "The burger was burnt.")),
        Row((5, "The service was slow. Waited 30 minutes for the server.")),
    ]
    df = ctx_with_model.read_rows(rows, schema)
    result = (
        df.map(
            prompt(
                "identify whether the {review} mentions waiting 20+ minutes to be "
                "acknowledged",
                bool,
            ).alias("is_long_wait")
        )
        .aggregate([bool_or(col("is_long_wait")).alias("exists_long_wait")])
        .check(col("exists_long_wait"))
        .collect()
    )

    validate_check_result(result, True, prov)


@pytest.mark.parametrize(
    "early_stop_enabled,prov",
    [
        pytest.param(
            False,
            Prov.pos_token((1,), col("has_sewer_problems"))
            * Prov.pos_token((5,), col("has_sewer_problems")),
            id="without_early_stop",
        ),
        pytest.param(
            True,
            Prov.pos_token((1,), col("has_sewer_problems"))
            * Prov.pos_token((5,), col("has_sewer_problems")),
            id="with_early_stop",
        ),
    ],
)
def test_cardinal_claim(
    ctx_with_model: SessionContext, early_stop_enabled: bool, prov: Prov
):
    if early_stop_enabled:
        ctx_with_model.enable_early_stop()

    schema = Schema((Field("id", int), Field("review", str)), ("id",))
    rows = [
        Row((1, "The restaurant had a weird smell. Kinda like a sewer!")),
        Row((2, "The service was fast. I only waited for 1 minute.")),
        Row((3, "I had to wait so long (~20 min) to get the server's attention.")),
        Row((4, "The burger was burnt.")),
        Row((5, "It smelled terrible. I had to leave. Like poop!")),
    ]
    df = ctx_with_model.read_rows(rows, schema)
    result = (
        df.map(
            prompt("identify whether the {review} mentions sewer smell problems").alias(
                "has_sewer_problems"
            )
        )
        .aggregate([count_if(col("has_sewer_problems")).alias("num_sewer_problems")])
        .check(col("num_sewer_problems") >= 2)
        .collect()
    )

    validate_check_result(result, True, prov)


@pytest.mark.parametrize(
    "early_stop_enabled,prov",
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
            id="without_early_stop",
        ),
        pytest.param(
            True,
            # Since the optimizer does not insert CountScan when a semantic
            # filter is below the aggregate, the aggregate proportion aggregation
            # cannot stop early. Thus, the full provenance is returned (assuming
            # minimal provenance is disabled).
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
            id="with_early_stop",
        ),
    ],
)
def test_proportional_claim(
    ctx_with_model: SessionContext, early_stop_enabled: bool, prov: Prov
):
    if early_stop_enabled:
        ctx_with_model.enable_early_stop()

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
    df = ctx_with_model.read_rows(rows, schema)
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


@pytest.mark.parametrize(
    "early_stop_enabled,prov",
    [
        pytest.param(
            False,
            Prov.neg_token((2,), col("is_six_dollars"))
            + Prov.neg_token((4,), col("is_six_dollars")),
            id="without_early_stop",
        ),
        pytest.param(
            True,
            Prov.neg_token((2,), col("is_six_dollars")),
            id="with_early_stop",
        ),
    ],
)
def test_universal_claim(
    ctx_with_model: SessionContext, early_stop_enabled: bool, prov: Prov
):
    if early_stop_enabled:
        ctx_with_model.enable_early_stop()

    schema = Schema((Field("id", int), Field("review", str)), ("id",))
    rows = [
        Row((1, "The restaurant had a fishy smell. Kinda like a sewer!")),
        Row((2, "They have $6 drinks and $5 dollar appetizers during happy hour.")),
        Row((3, "Come during happy hour! Six dollars for an appetizer.")),
        Row((4, "Come during happy hour! Seven dollars for an appetizer.")),
        Row((5, "I ordered the tuna melt. It was the best in town.")),
    ]
    df = ctx_with_model.read_rows(rows, schema)
    result = (
        df.filter(prompt("the {review} mentions the price for happy hour appetizers"))
        .map(
            prompt(
                "identify whether the {review} says that the price of happy hour "
                "appetizers is $6"
            ).alias("is_six_dollars")
        )
        .aggregate([bool_and(col("is_six_dollars")).alias("all_six_dollars")])
        .check(col("all_six_dollars"))
        .collect()
    )

    validate_check_result(result, False, prov)
