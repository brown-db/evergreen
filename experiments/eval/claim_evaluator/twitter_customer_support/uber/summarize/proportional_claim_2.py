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
                    "Identify whether in the {dialog} the customer received "
                    "inconsistent or inaccurate information from the support agent",
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
            .check(col("inconsistent_info_prop") < 0.40)
        )

    def check_predicate(self) -> Expr:
        return col("inconsistent_info_prop") < 0.40

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("received_inconsistent_info"),)

    def hints(self) -> str:
        return ""


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = ProportionalClaim2Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/twitter_customer_support/uber/summarize/proportional_claim_2_2026-06-01_19-55-39.json"
        ),
        dataset_key=("dialog_id",),
        text_field_name="dialog",
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
