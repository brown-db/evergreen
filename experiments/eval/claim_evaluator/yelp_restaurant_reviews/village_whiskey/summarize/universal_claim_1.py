from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import Expr, bool_and, col, prompt
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import REVIEW_SCHEMA


class UniversalClaim1Evaluator(ClaimEvaluator):
    def reference_query(
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
            .filter(
                prompt("The {text} mentions the number of whiskey varieties available")
            )
            .map(
                prompt(
                    "Identify whether the {text} indicates that there are over 200 "
                    "varieties of whiskey available",
                    bool,
                ).alias("indicates_over_200")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [bool_and(col("indicates_over_200")).alias("all_confirm_over_200")]
            )
            .check(col("all_confirm_over_200"))
        )

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("indicates_over_200"),)

    def filter_prompt_str(self) -> str | None:
        return "The {text} mentions the number of whiskey varieties available"


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = UniversalClaim1Evaluator(
        name="universal_claim_1",
        claim="There are over 200 varieties of whiskey available at Village Whiskey.",
        hints=(
            "Interpret the claim as an implicit universal quantification over "
            "all reviews that mention the number of whiskey varieties available."
        ),
        schema=REVIEW_SCHEMA,
        text_field_name="text",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/yelp_restaurant_reviews/village_whiskey/summarize_2026-02-10_17-32-55.json"
        ),
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
