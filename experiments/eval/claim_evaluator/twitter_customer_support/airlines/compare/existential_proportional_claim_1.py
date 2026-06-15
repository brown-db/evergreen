from enum import Enum
from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import (
    bool_or,
    col,
    prompt,
    proportion,
)
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import DIALOG_WITH_COMPANY_SCHEMA


class Satisfaction(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    MIXED = "mixed"
    NEUTRAL = "neutral"


class ExistentialProportionalClaim1Evaluator(ClaimEvaluator):
    NAME = "existential_proportional_claim_1"
    CLAIM = "The customer satisfaction for some airlines was a mixed bag."
    HINTS = (
        "Classify each dialog's satisfaction as positive, negative, mixed (both), "
        "or neutral. An airline's satisfaction is a 'mixed bag' if at least 20% of "
        "its dialogs are positive or mixed and at least 20% are negative or mixed."
    )
    SCHEMA = DIALOG_WITH_COMPANY_SCHEMA
    TEXT_FIELD_NAME = "dialog"
    AGG_RESULT_PATH = Path(
        "experiments/results/semantic_aggregate/twitter_customer_support/airlines/compare_2026-06-01_14-04-34.json"
    )

    def reference_query(
        self,
        df: DataFrame,
        impl: Implementation,
        language_models: tuple[str, ...],
        trial_id: int,
    ) -> DataFrame:
        return (
            df.map(
                prompt(
                    "Based on the customer support {dialog}, identify the customer's "
                    "satisfaction level: "
                    "positive (customer seems satisfied/happy with the resolution), "
                    "negative (customer seems dissatisfied/frustrated), "
                    "mixed (customer expresses both satisfaction and dissatisfaction), "
                    "or neutral (no clear satisfaction signal)",
                    Satisfaction,
                ).alias("satisfaction")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [
                    proportion(
                        col("satisfaction").eq(Satisfaction.POSITIVE)
                        | col("satisfaction").eq(Satisfaction.MIXED)
                    ).alias("positive_or_mixed_prop"),
                    proportion(
                        col("satisfaction").eq(Satisfaction.NEGATIVE)
                        | col("satisfaction").eq(Satisfaction.MIXED)
                    ).alias("negative_or_mixed_prop"),
                ],
                group_by=[col("company_id")],
            )
            .aggregate(
                [
                    bool_or(
                        (col("positive_or_mixed_prop") >= 0.2)
                        & (col("negative_or_mixed_prop") >= 0.2)
                    ).alias("some_mixed_bag")
                ]
            )
            .check(col("some_mixed_bag"))
        )


if __name__ == "__main__":
    ExistentialProportionalClaim1Evaluator().evaluate(parse_claim_evaluator_args())
