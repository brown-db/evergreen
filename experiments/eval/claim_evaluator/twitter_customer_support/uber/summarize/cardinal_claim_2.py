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
    CLAIM = "Less than 725 customers complained about poor driver behavior."
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
                    "Identify whether the customer support {dialog} contains a "
                    "customer complaint about poor driver behavior",
                    bool,
                ).alias("complains_about_poor_driver_behavior")
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
                    count_if(col("complains_about_poor_driver_behavior")).alias(
                        "poor_driver_complaint_count"
                    )
                ]
            )
            .check(col("poor_driver_complaint_count") < 725)
        )


if __name__ == "__main__":
    CardinalClaim2Evaluator().evaluate(parse_claim_evaluator_args())
