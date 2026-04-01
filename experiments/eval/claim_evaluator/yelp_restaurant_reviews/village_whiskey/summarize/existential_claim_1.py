from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import Expr, bool_or, col, prompt
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)


class ExistentialClaim1Evaluator(ClaimEvaluator):
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
                    "Identify whether the {text} expresses enjoyment of the "
                    "restaurant's bacon fries",
                    bool,
                ).alias("enjoys_bacon_fries")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [bool_or(col("enjoys_bacon_fries")).alias("some_enjoy_bacon_fries")]
            )
            .check(col("some_enjoy_bacon_fries"))
        )

    def check_predicate(self) -> Expr:
        return col("some_enjoy_bacon_fries")

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("enjoys_bacon_fries"),)

    def hints(self) -> str:
        return ""


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = ExistentialClaim1Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/yelp_restaurant_reviews/village_whiskey/summarize/existential_claim_1_2026-02-16_14-46-14.json"
        ),
        dataset_key=("review_id",),
        text_field_name="text",
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
