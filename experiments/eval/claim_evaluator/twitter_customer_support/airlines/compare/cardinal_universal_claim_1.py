from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import (
    Expr,
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


class CardinalUniversalClaim1Evaluator(ClaimEvaluator):
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
                    "Identify whether the support agent apologizes in the {dialog}",
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

    def check_predicate(self) -> Expr:
        return col("num_companies_all_apologies").eq(2)

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("agent_apologizes"),)

    def hints(self) -> str:
        return ""

    def filter_prompt_str(self) -> str | None:
        return "The {dialog} contains a customer complaint"


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = CardinalUniversalClaim1Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/twitter_customer_support/airlines/compare/cardinal_universal_claim_1_2026-06-02_15-21-36.json"
        ),
        dataset_key=("dialog_id",),
        text_field_name="dialog",
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
