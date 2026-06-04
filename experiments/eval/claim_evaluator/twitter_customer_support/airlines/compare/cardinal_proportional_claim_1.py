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


class CardinalProportionalClaim1Evaluator(ClaimEvaluator):
    def query(
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
                    count_if(col("delay_complaint_prop") > 0.10).alias(
                        "num_companies_with_delay_complaints"
                    )
                ]
            )
            .check(col("num_companies_with_delay_complaints").eq(4))
        )

    def check_predicate(self) -> Expr:
        return col("num_companies_with_delay_complaints").eq(4)

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("about_flight_delays"),)

    def hints(self) -> str:
        return ""

    def filter_prompt_str(self) -> str | None:
        return "The {dialog} contains a customer complaint"


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = CardinalProportionalClaim1Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/twitter_customer_support/airlines/compare/cardinal_proportional_claim_1_2026-06-02_14-17-13.json"
        ),
        dataset_key=("dialog_id",),
        text_field_name="dialog",
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
