from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import Expr, bool_or, col, prompt, proportion
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import REVIEW_WITH_BUSINESS_SCHEMA


class ProportionalExistentialClaim1Evaluator(ClaimEvaluator):
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

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("mentions_cold_food"),)


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = ProportionalExistentialClaim1Evaluator(
        name="proportional_existential_claim_1",
        claim="The majority of McDonald's locations had reports of cold food.",
        hints="",
        schema=REVIEW_WITH_BUSINESS_SCHEMA,
        text_field_name="text",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/yelp_restaurant_reviews/mcdonalds_mo/compare_2026-02-10_19-11-51.json"
        ),
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
