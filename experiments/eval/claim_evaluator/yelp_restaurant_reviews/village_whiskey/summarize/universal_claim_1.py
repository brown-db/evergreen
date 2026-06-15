from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import bool_and, col, prompt
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import REVIEW_SCHEMA


class UniversalClaim1Evaluator(ClaimEvaluator):
    NAME = "universal_claim_1"
    CLAIM = "There are over 200 varieties of whiskey available at Village Whiskey."
    HINTS = (
        "Interpret the claim as an implicit universal quantification over "
        "all reviews that mention the number of whiskey varieties available."
    )
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
            df.log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.PRE_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .filter(
                prompt(
                    "The restaurant review {text} mentions the number of whiskey "
                    "varieties available"
                )
            )
            .map(
                prompt(
                    "Identify whether the restaurant review {text} indicates that "
                    "there are over 200 varieties of whiskey available",
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


if __name__ == "__main__":
    UniversalClaim1Evaluator().evaluate(parse_claim_evaluator_args())
