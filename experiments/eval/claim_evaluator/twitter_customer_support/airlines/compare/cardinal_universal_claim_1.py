from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import (
    bool_and,
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


class CardinalUniversalClaim1Evaluator(ClaimEvaluator):
    NAME = "cardinal_universal_claim_1"
    CLAIM = (
        "There are 2 airline companies where the support agent apologizes in all "
        "dialogs with a customer complaint."
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
                    "Identify whether the support agent apologizes in the customer "
                    "support {dialog}",
                    bool,
                ).alias("agent_apologizes")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [bool_and(col("agent_apologizes")).alias("all_apologies")],
                group_by=[col("company_id")],
            )
            .aggregate(
                [count_if(col("all_apologies")).alias("num_companies_all_apologies")]
            )
            .check(col("num_companies_all_apologies").eq(2))
        )


if __name__ == "__main__":
    CardinalUniversalClaim1Evaluator().evaluate(parse_claim_evaluator_args())
