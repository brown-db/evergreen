from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import (
    Expr,
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


class ProportionalClaim2Evaluator(ClaimEvaluator):
    def query(
        self,
        df: DataFrame,
        impl: Implementation,
        language_models: tuple[str, ...],
        trial_id: int,
    ) -> DataFrame:
        return (
            df.map(
                prompt(
                    "Identify whether the {dialog} involves a customer experiencing "
                    "problems with payments or billing",
                    bool,
                ).alias("has_payment_billing_problem")
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
                    proportion(col("has_payment_billing_problem")).alias(
                        "payment_billing_prop"
                    )
                ]
            )
            .check(col("payment_billing_prop").ne(0.10))
        )

    def check_predicate(self) -> Expr:
        return col("payment_billing_prop").ne(0.10)

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("has_payment_billing_problem"),)

    def hints(self) -> str:
        return ""


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = ProportionalClaim2Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/twitter_customer_support/play_station/summarize/proportional_claim_2_2026-06-01_16-09-34.json"
        ),
        dataset_key=("dialog_id",),
        text_field_name="dialog",
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
