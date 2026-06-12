from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import (
    Expr,
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
            .filter(prompt("The {dialog} contains a customer complaint"))
            .map(
                prompt(
                    "Identify whether the {dialog} contains a customer complaint "
                    "regarding flight delays",
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

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("about_flight_delays"),)

    def filter_prompt_str(self) -> str | None:
        return "The {dialog} contains a customer complaint"


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = CardinalProportionalClaim1Evaluator(
        name="cardinal_proportional_claim_1",
        claim="There are 4 airline companies where over 15% of customer complaints are "
        "regarding flight delays.",
        hints="",
        schema=DIALOG_WITH_COMPANY_SCHEMA,
        text_field_name="dialog",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/twitter_customer_support/airlines/compare_2026-06-01_14-04-34.json"
        ),
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
