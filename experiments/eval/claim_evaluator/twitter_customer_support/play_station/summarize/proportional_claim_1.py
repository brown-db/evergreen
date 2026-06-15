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
from experiments.schemas import DIALOG_SCHEMA


class ProportionalClaim1Evaluator(ClaimEvaluator):
    NAME = "proportional_claim_1"
    CLAIM = "10% of customers are experiencing problems with payments or billing."
    SCHEMA = DIALOG_SCHEMA
    TEXT_FIELD_NAME = "dialog"
    AGG_RESULT_PATH = Path(
        "experiments/results/semantic_aggregate/twitter_customer_support/play_station/summarize_2026-06-01_14-04-42.json"
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
                    "Identify whether the customer support {dialog} involves a "
                    "customer experiencing problems with payments or billing",
                    bool,
                ).alias("has_payment_billing_issue")
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
                    proportion(col("has_payment_billing_issue")).alias(
                        "payment_billing_prop"
                    )
                ]
            )
            .check(col("payment_billing_prop").eq(0.10))
        )


if __name__ == "__main__":
    ProportionalClaim1Evaluator().evaluate(parse_claim_evaluator_args())
