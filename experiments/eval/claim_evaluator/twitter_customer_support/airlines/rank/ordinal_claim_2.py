from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import (
    col,
    count_if,
    prompt,
)
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import DIALOG_WITH_COMPANY_SCHEMA


class OrdinalClaim2Evaluator(ClaimEvaluator):
    NAME = "ordinal_claim_2"
    CLAIM = (
        "SouthwestAir has the second most customer support dialogs with "
        "complaints about flight booking issues."
    )
    SCHEMA = DIALOG_WITH_COMPANY_SCHEMA
    TEXT_FIELD_NAME = "dialog"
    AGG_RESULT_PATH = Path(
        "experiments/results/semantic_aggregate/twitter_customer_support/airlines/rank_2026-06-03_18-22-36.json"
    )
    CACHE_ID = "ordinal_claims_1_and_2"

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
                    "Identify whether the customer support {dialog} contains a "
                    "customer complaint about flight booking issues",
                    bool,
                ).alias("has_booking_complaint")
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
                    count_if(col("has_booking_complaint")).alias(
                        "booking_complaint_count"
                    )
                ],
                group_by=[col("company_id")],
            )
            .with_rank(col("booking_complaint_count"))
            .filter(col("company_id").eq("SouthwestAir"))
            .check(col("rank").eq(2))
        )


if __name__ == "__main__":
    OrdinalClaim2Evaluator().evaluate(parse_claim_evaluator_args())
