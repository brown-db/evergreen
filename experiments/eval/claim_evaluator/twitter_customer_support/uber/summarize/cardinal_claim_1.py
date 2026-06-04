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


class CardinalClaim1Evaluator(ClaimEvaluator):
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
                    "Identify whether the customer in this {dialog} complains about "
                    "poor driver behavior (e.g., rude, unprofessional, unsafe driving, "
                    "or other negative driver conduct)",
                    bool,
                ).alias("complains_poor_driver_behavior")
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
                    count_if(col("complains_poor_driver_behavior")).alias(
                        "poor_driver_count"
                    )
                ]
            )
            .check(col("poor_driver_count").eq(801))
        )

    def check_predicate(self) -> Expr:
        return col("poor_driver_count").eq(801)

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("complains_poor_driver_behavior"),)

    def hints(self) -> str:
        return ""


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = CardinalClaim1Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/twitter_customer_support/uber/summarize/cardinal_claim_1_2026-06-01_22-05-35.json"
        ),
        dataset_key=("dialog_id",),
        text_field_name="dialog",
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
