from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import Expr, bool_or, col, prompt, proportion
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)


class ProportionalExistentialClaim1Evaluator(ClaimEvaluator):
    def query(
        self,
        df: DataFrame,
        impl: Implementation,
        language_models: tuple[str, ...],
        trial_id: int,
    ) -> DataFrame:
        return (
            df.map(
                prompt(
                    "Identify whether the {text} mentions or complains about cold food",
                    bool,
                ).alias("mentions_cold_food")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [bool_or(col("mentions_cold_food")).alias("has_cold_food_report")],
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

    def check_predicate(self) -> Expr:
        return col("prop_locations_with_cold_food") > 0.5

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("mentions_cold_food"),)

    def hints(self) -> str:
        return ""


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = ProportionalExistentialClaim1Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/yelp_restaurant_reviews/mcdonalds_mo/compare/proportional_existential_claim_1_2026-02-16_23-41-52.json"
        ),
        dataset_key=("review_id",),
        text_field_name="text",
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
