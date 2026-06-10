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
from experiments.schemas import DIALOG_SCHEMA


class ProportionalClaim2Evaluator(ClaimEvaluator):
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
                    "Identify whether the {dialog} involves a customer experiencing "
                    "problems with payments or billing",
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
            .check(col("payment_billing_prop").ne(0.10))
        )

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("has_payment_billing_issue"),)


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = ProportionalClaim2Evaluator(
        name="proportional_claim_2",
        claim="The percentage of customers experiencing problems with payments or "
        "billing is not 10%.",
        hints="",
        schema=DIALOG_SCHEMA,
        text_field_name="dialog",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/twitter_customer_support/play_station/summarize_2026-06-01_14-04-42.json"
        ),
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
