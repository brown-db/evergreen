from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import col, count_if, prompt, proportion
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import REVIEW_WITH_BUSINESS_SCHEMA


class ProportionalCardinalClaim1Evaluator(ClaimEvaluator):
    NAME = "proportional_cardinal_claim_1"
    CLAIM = (
        "Only a minority of McDonald's locations had multiple reports of "
        "incorrect orders."
    )
    SCHEMA = REVIEW_WITH_BUSINESS_SCHEMA
    TEXT_FIELD_NAME = "text"
    AGG_RESULT_PATH = Path(
        "experiments/results/semantic_aggregate/yelp_restaurant_reviews/mcdonalds_mo/compare_2026-02-10_19-11-51.json"
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
                    "Identify whether the restaurant review {text} reports an "
                    "incorrect order",
                    bool,
                ).alias("reports_incorrect_order")
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
                    count_if(col("reports_incorrect_order")).alias(
                        "incorrect_order_count"
                    )
                ],
                group_by=[col("business_id")],
            )
            .aggregate(
                [
                    proportion(col("incorrect_order_count") >= 2).alias(
                        "prop_with_multiple_incorrect"
                    )
                ]
            )
            .check(col("prop_with_multiple_incorrect") < 0.5)
        )


if __name__ == "__main__":
    ProportionalCardinalClaim1Evaluator().evaluate(parse_claim_evaluator_args())
