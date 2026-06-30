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
    CLAIM = "Common criticisms of John's Roast Pork include cash-only policy."
    HINTS = (
        "The phrase 'common criticisms' suggests a threshold of at least 10% "
        "of the reviews that express criticism."
    )
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
            df.log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.PRE_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .filter(
                prompt(
                    "The restaurant review {text} expresses a complaint about the "
                    "restaurant"
                )
            )
            .map(
                prompt(
                    "Identify whether the restaurant review {text} complains about the "
                    "restaurant's cash-only policy",
                    bool,
                ).alias("complains_about_cash_only")
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
                    proportion(col("complains_about_cash_only")).alias(
                        "cash_only_complaint_prop"
                    )
                ]
            )
            .check(col("cash_only_complaint_prop") >= 0.1)
        )


if __name__ == "__main__":
    ProportionalClaim1Evaluator().evaluate(parse_claim_evaluator_args())
