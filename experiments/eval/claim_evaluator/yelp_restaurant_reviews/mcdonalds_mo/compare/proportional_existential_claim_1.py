from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import bool_or, col, prompt, proportion
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import REVIEW_WITH_BUSINESS_SCHEMA


class ProportionalExistentialClaim1Evaluator(ClaimEvaluator):
    NAME = "proportional_existential_claim_1"
    CLAIM = "The majority of McDonald's locations had reports of cold food."
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
                    "Identify whether the restaurant review {text} reports about "
                    "cold food",
                    bool,
                ).alias("reports_cold_food")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [bool_or(col("reports_cold_food")).alias("has_cold_food_report")],
                group_by=[col("business_id")],
            )
            .aggregate(
                [
                    proportion(col("has_cold_food_report")).alias(
                        "prop_locations_with_cold_food"
                    )
                ]
            )
            .check(col("prop_locations_with_cold_food") > 0.5)
        )


if __name__ == "__main__":
    ProportionalExistentialClaim1Evaluator().evaluate(parse_claim_evaluator_args())
