from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import bool_or, col, prompt
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import REVIEW_SCHEMA


class ExistentialClaim1Evaluator(ClaimEvaluator):
    NAME = "existential_claim_1"
    CLAIM = "Some reviewers enjoy the restaurant's chicken salad."
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
                    "Identify whether the restaurant review {text} expresses enjoyment "
                    "of the restaurant's chicken salad",
                    bool,
                ).alias("enjoys_chicken_salad")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [bool_or(col("enjoys_chicken_salad")).alias("some_enjoy_chicken_salad")]
            )
            .check(col("some_enjoy_chicken_salad"))
        )


if __name__ == "__main__":
    ExistentialClaim1Evaluator().evaluate(parse_claim_evaluator_args())
