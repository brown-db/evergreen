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


class OrdinalClaim2Evaluator(ClaimEvaluator):
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
                    "Identify whether the {dialog} contains a complaint about flight "
                    "booking issues",
                    bool,
                ).alias("has_booking_complaint")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [count_if(col("has_booking_complaint")).alias("complaint_count")],
                group_by=[col("company_id")],
            )
            .with_rank(col("complaint_count"))
            .filter(col("company_id").eq("SouthwestAir"))
            .check(col("rank").eq(2))
        )

    def check_predicate(self) -> Expr:
        return col("rank").eq(2)

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("has_booking_complaint"),)

    def hints(self) -> str:
        return ""


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = OrdinalClaim2Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/twitter_customer_support/airlines/rank/ordinal_claim_2_2026-06-04_16-43-15.json"
        ),
        dataset_key=("dialog_id",),
        text_field_name="dialog",
        random_seed=42,
        cache_id="ordinal_claims_1_and_2",
    )
    claim_evaluator.evaluate(args)
