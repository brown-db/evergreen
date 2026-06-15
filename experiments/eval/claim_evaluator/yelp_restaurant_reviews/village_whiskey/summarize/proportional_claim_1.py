from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import (
    col,
    prompt,
    proportion,
)
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import REVIEW_SCHEMA


class ProportionalClaim1Evaluator(ClaimEvaluator):
    NAME = "proportional_claim_1"
    CLAIM = "Village Whiskey has received majority positive reviews."
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
                    "Identify whether the restaurant review {text} expresses a "
                    "positive sentiment towards the restaurant",
                    bool,
                ).alias("is_positive")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate([proportion(col("is_positive")).alias("positive_prop")])
            .check(col("positive_prop") > 0.5)
        )


if __name__ == "__main__":
    ProportionalClaim1Evaluator().evaluate(parse_claim_evaluator_args())
