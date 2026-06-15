from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import Expr, col, count_if, prompt
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import REVIEW_SCHEMA


class CardinalClaim1Evaluator(ClaimEvaluator):
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
                    "Identify whether the restaurant review {text} is from a "
                    "vegetarian reviewer who enjoyed the restaurant's burgers",
                    bool,
                ).alias("vegetarian_enjoyed_burgers")
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
                    count_if(col("vegetarian_enjoyed_burgers")).alias(
                        "vegetarian_burger_enjoyers"
                    )
                ]
            )
            .check(col("vegetarian_burger_enjoyers") > 5)
        )

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("vegetarian_enjoyed_burgers"),)


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = CardinalClaim1Evaluator(
        name="cardinal_claim_1",
        claim="More than a handful of vegetarian customers enjoyed the restaurant's "
        "burgers.",
        hints=(
            "The phrase 'more than a handful' suggests a threshold of greater than 5."
        ),
        schema=REVIEW_SCHEMA,
        text_field_name="text",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/yelp_restaurant_reviews/village_whiskey/summarize_2026-02-10_17-32-55.json"
        ),
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
