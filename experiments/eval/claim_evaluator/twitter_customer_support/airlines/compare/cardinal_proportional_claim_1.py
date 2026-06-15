from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import (
    col,
    count_if,
    prompt,
    proportion,
)
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import DIALOG_WITH_COMPANY_SCHEMA


class CardinalProportionalClaim1Evaluator(ClaimEvaluator):
    NAME = "cardinal_proportional_claim_1"
    CLAIM = (
        "There are 4 airline companies where over 15% of customer complaints are "
        "regarding flight delays."
    )
    SCHEMA = DIALOG_WITH_COMPANY_SCHEMA
    TEXT_FIELD_NAME = "dialog"
    AGG_RESULT_PATH = Path(
        "experiments/results/semantic_aggregate/twitter_customer_support/airlines/compare_2026-06-01_14-04-34.json"
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
                prompt("The customer support {dialog} contains a customer complaint")
            )
            .map(
                prompt(
                    "Identify whether the customer support {dialog} contains a "
                    "customer complaint regarding flight delays",
                    bool,
                ).alias("about_flight_delays")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [proportion(col("about_flight_delays")).alias("delay_complaint_prop")],
                group_by=[col("company_id")],
            )
            .aggregate(
                [
                    count_if(col("delay_complaint_prop") > 0.15).alias(
                        "num_companies_with_delay_complaints"
                    )
                ]
            )
            .check(col("num_companies_with_delay_complaints").eq(4))
        )


if __name__ == "__main__":
    CardinalProportionalClaim1Evaluator().evaluate(parse_claim_evaluator_args())
