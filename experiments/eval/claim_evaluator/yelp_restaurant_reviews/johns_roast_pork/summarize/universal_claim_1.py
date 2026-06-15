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
    CLAIM = "The reviews do not mention any vegan options at the restaurant."
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
                    "Identify whether the restaurant review {text} mentions the "
                    "restaurant's vegan options",
                    bool,
                ).alias("mentions_vegan_options")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [bool_and(~col("mentions_vegan_options")).alias("none_mention_vegan")]
            )
            .check(col("none_mention_vegan"))
        )


if __name__ == "__main__":
    UniversalClaim1Evaluator().evaluate(parse_claim_evaluator_args())
