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
from experiments.schemas import DIALOG_WITH_COMPANY_SCHEMA


class UniversalExistentialClaim1Evaluator(ClaimEvaluator):
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
                    "the taste of the food provided on their flight",
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

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("complains_about_food"),)


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = UniversalExistentialClaim1Evaluator(
        name="universal_existential_claim_1",
        claim="All airlines have customer complaints about the taste of the food "
        "provided on their flight.",
        hints="",
        schema=DIALOG_WITH_COMPANY_SCHEMA,
        text_field_name="dialog",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/twitter_customer_support/airlines/compare_2026-06-01_14-04-34.json"
        ),
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
