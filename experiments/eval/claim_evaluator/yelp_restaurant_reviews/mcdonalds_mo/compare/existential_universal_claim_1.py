from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import (
    Expr,
    bool_and,
    bool_or,
    col,
    prompt,
)
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)


class ExistentialUniversalClaim1Evaluator(ClaimEvaluator):
    def query(
        self,
        df: DataFrame,
        impl: Implementation,
        language_models: tuple[str, ...],
        trial_id: int,
    ) -> DataFrame:
        return (
            df.map(
                prompt("Identify whether the {text} is a negative review", bool).alias(
                    "is_negative"
                )
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [bool_and(~col("is_negative")).alias("has_no_negative_reviews")],
                group_by=[col("business_id")],
            )
            .aggregate(
                [
                    bool_or(col("has_no_negative_reviews")).alias(
                        "some_locations_no_negative"
                    )
                ]
            )
            .check(col("some_locations_no_negative"))
        )

    def check_predicate(self) -> Expr:
        return col("some_locations_no_negative")

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("is_negative"),)

    def hints(self) -> str:
        return ""


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = ExistentialUniversalClaim1Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/yelp_restaurant_reviews/mcdonalds_mo/compare/existential_universal_claim_1_2026-02-18_20-28-39.json"
        ),
        dataset_key=("review_id",),
        text_field_name="text",
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
