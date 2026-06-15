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
from experiments.schemas import DIALOG_SCHEMA


class CardinalClaim2Evaluator(ClaimEvaluator):
    NAME = "cardinal_claim_2"
    CLAIM = (
        "The number of customers experiencing problems with their accounts or "
        "login issues is not 240."
    )
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
                    "customer experiencing problems with their account or login issues",
                    bool,
                ).alias("has_account_or_login_issue")
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
                    count_if(col("has_account_or_login_issue")).alias(
                        "account_login_count"
                    )
                ]
            )
            .check(col("account_login_count").ne(240))
        )


if __name__ == "__main__":
    CardinalClaim2Evaluator().evaluate(parse_claim_evaluator_args())
