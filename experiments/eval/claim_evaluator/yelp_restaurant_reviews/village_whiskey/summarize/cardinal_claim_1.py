from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import col, count_if, prompt
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import REVIEW_SCHEMA


class CardinalClaim1Evaluator(ClaimEvaluator):
    NAME = "cardinal_claim_1"
    CLAIM = (
        "More than a handful of vegetarian customers enjoyed the restaurant's "
        "burgers."
    )
    HINTS = "The phrase 'more than a handful' suggests a threshold of greater than 5."
    SCHEMA = REVIEW_SCHEMA
    TEXT_FIELD_NAME = "text"
    AGG_RESULT_PATH = Path(
        "experiments/results/semantic_aggregate/yelp_restaurant_reviews/village_whiskey/summarize_2026-02-10_17-32-55.json"
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


if __name__ == "__main__":
    CardinalClaim1Evaluator().evaluate(parse_claim_evaluator_args())
