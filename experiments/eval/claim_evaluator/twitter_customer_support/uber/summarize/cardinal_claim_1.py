from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import (
    Expr,
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


class CardinalClaim1Evaluator(ClaimEvaluator):
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
                    "Identify whether the {dialog} contains a customer complaint about "
                    "poor driver behavior",
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
            .check(col("poor_driver_complaint_count").eq(801))
        )

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("complains_about_poor_driver_behavior"),)


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = CardinalClaim1Evaluator(
        name="cardinal_claim_1",
        claim="801 customers complained about poor driver behavior.",
        hints="",
        schema=DIALOG_SCHEMA,
        text_field_name="dialog",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/twitter_customer_support/uber/summarize_2026-06-01_14-04-38.json"
        ),
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
