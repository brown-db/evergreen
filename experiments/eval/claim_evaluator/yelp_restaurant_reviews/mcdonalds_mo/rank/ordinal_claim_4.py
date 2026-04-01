from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import Expr, col, prompt, proportion
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)


class OrdinalClaim4Evaluator(ClaimEvaluator):
    def query(
        self,
        df: DataFrame,
        impl: Implementation,
        language_models: tuple[str, ...],
        trial_id: int,
    ) -> DataFrame:
        return (
            df.log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.PRE_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .filter(prompt("The {text} mentions the service at the restaurant"))
            .map(
                prompt(
                    "Identify whether the {text} praises or speaks positively about "
                    "the service at the restaurant",
                    bool,
                ).alias("praises_service")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [proportion(col("praises_service")).alias("service_prop")],
                group_by=[col("business_id")],
            )
            .with_rank(col("service_prop"))
            .filter(col("business_id").eq("XlbFKW_Keun8qh1S5m3QgQ"))
            .check(col("rank").eq(2))
        )

    def check_predicate(self) -> Expr:
        return col("rank").eq(2)

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("praises_service"),)

    def hints(self) -> str:
        return (
            "Rank based on the proportion of reviews that speak positively about "
            "the service at each restaurant."
        )


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = OrdinalClaim4Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/yelp_restaurant_reviews/mcdonalds_mo/rank/ordinal_claim_4_2026-02-22_11-38-59.json"
        ),
        dataset_key=("review_id",),
        text_field_name="text",
        random_seed=42,
        cache_id="ordinal_claims_3_and_4",
    )
    claim_evaluator.evaluate(args)
