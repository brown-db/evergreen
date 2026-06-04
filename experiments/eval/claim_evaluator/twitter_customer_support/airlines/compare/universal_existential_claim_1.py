from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import (
    Expr,
    bool_and,
    bool_or,
    col,
    prompt,
)
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)


class UniversalExistentialClaim1Evaluator(ClaimEvaluator):
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
                    "Identify whether the {dialog} contains a customer complaint about "
                    "the airplane's food",
                    bool,
                ).alias("complains_about_food")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [bool_or(col("complains_about_food")).alias("has_food_complaint")],
                group_by=[col("company_id")],
            )
            .aggregate(
                [bool_and(col("has_food_complaint")).alias("all_have_food_complaints")]
            )
            .check(col("all_have_food_complaints"))
        )

    def check_predicate(self) -> Expr:
        return col("all_have_food_complaints")

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("complains_about_food"),)

    def hints(self) -> str:
        return ""


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = UniversalExistentialClaim1Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/twitter_customer_support/airlines/compare/universal_existential_claim_1_2026-06-03_14-38-26.json"
        ),
        dataset_key=("dialog_id",),
        text_field_name="dialog",
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
