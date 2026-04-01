from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import Expr, col, count_if, prompt
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)


class CardinalClaim1Evaluator(ClaimEvaluator):
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
            .filter(
                prompt(
                    "The {text} indicates that the reviewer is a vegetarian or "
                    "identifies as vegetarian"
                )
            )
            .map(
                prompt(
                    "Identify whether the {text} indicates that the reviewer enjoyed "
                    "the restaurant's burgers",
                    bool,
                ).alias("enjoyed_burgers")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [count_if(col("enjoyed_burgers")).alias("vegetarian_burger_enjoyers")]
            )
            .check(col("vegetarian_burger_enjoyers") > 5)
        )

    def check_predicate(self) -> Expr:
        return col("vegetarian_burger_enjoyers") > 5

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("enjoyed_burgers"),)

    def hints(self) -> str:
        return (
            "The phrase 'more than a handful' suggests a threshold of greater than 5."
        )

    def filter_prompt_str(self) -> str | None:
        return (
            "The {text} indicates that the reviewer is a vegetarian or "
            "identifies as vegetarian"
        )


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = CardinalClaim1Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/yelp_restaurant_reviews/village_whiskey/summarize/cardinal_claim_1_2026-02-16_11-46-50.json"
        ),
        dataset_key=("review_id",),
        text_field_name="text",
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
