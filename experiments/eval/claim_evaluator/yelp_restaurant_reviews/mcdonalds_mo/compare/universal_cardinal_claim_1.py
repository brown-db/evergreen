from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import Expr, bool_and, col, count_if, prompt
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)


class UniversalCardinalClaim1Evaluator(ClaimEvaluator):
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
                    "Identify whether the {text} is a complaint about poor service "
                    "quality",
                    bool,
                ).alias("complains_about_service")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [count_if(col("complains_about_service")).alias("complaint_count")],
                group_by=[col("business_id")],
            )
            .aggregate(
                [
                    bool_and(col("complaint_count") >= 2).alias(
                        "all_have_multiple_complaints"
                    )
                ]
            )
            .check(col("all_have_multiple_complaints"))
        )

    def check_predicate(self) -> Expr:
        return col("all_have_multiple_complaints")

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("complains_about_service"),)

    def hints(self) -> str:
        return ""


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = UniversalCardinalClaim1Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/yelp_restaurant_reviews/mcdonalds_mo/compare/universal_cardinal_claim_1_2026-02-17_01-01-32.json"
        ),
        dataset_key=("review_id",),
        text_field_name="text",
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
