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
        "Less than a handful of customers complained about the restaurant's "
        "service quality."
    )
    HINTS = "The phrase 'less than a handful' suggests a threshold of less than 5."
    SCHEMA = REVIEW_SCHEMA
    TEXT_FIELD_NAME = "text"
    AGG_RESULT_PATH = Path(
        "experiments/results/semantic_aggregate/yelp_restaurant_reviews/johns_roast_pork/summarize_2026-02-10_18-24-08.json"
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
                    "Identify whether the restaurant review {text} complains about the "
                    "restaurant's service quality",
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
                [
                    count_if(col("complains_about_service")).alias(
                        "service_complaint_count"
                    )
                ]
            )
            .check(col("service_complaint_count") < 5)
        )


if __name__ == "__main__":
    CardinalClaim1Evaluator().evaluate(parse_claim_evaluator_args())
