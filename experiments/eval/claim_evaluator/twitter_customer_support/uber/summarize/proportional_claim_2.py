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


class ProportionalClaim2Evaluator(ClaimEvaluator):
    NAME = "proportional_claim_2"
    CLAIM = (
        "Less than 25% of customers reported receiving inconsistent or "
        "inaccurate information from support agents."
    )
    SCHEMA = DIALOG_SCHEMA
    TEXT_FIELD_NAME = "dialog"
    AGG_RESULT_PATH = Path(
        "experiments/results/semantic_aggregate/twitter_customer_support/uber/summarize_2026-06-01_14-04-38.json"
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
                    "Identify whether the customer support {dialog} indicates that the "
                    "customer received inconsistent or inaccurate information from the "
                    "support agent",
                    bool,
                ).alias("received_inconsistent_info")
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
                    proportion(col("received_inconsistent_info")).alias(
                        "inconsistent_info_prop"
                    )
                ]
            )
            .check(col("inconsistent_info_prop") < 0.25)
        )


if __name__ == "__main__":
    ProportionalClaim2Evaluator().evaluate(parse_claim_evaluator_args())
