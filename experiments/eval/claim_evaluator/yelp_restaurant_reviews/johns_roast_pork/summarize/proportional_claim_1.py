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


class ProportionalClaim1Evaluator(ClaimEvaluator):
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
            .filter(prompt("The {text} expresses a criticism or complaint"))
            .map(
                prompt(
                    "Identify whether the {text} criticizes or complains about a "
                    "cash-only policy",
                    bool,
                ).alias("criticizes_cash_only")
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
                    proportion(col("criticizes_cash_only")).alias(
                        "cash_only_criticism_prop"
                    )
                ]
            )
            .check(col("cash_only_criticism_prop") >= 0.1)
        )

    def check_predicate(self) -> Expr:
        return col("cash_only_criticism_prop") >= 0.1

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("criticizes_cash_only"),)

    def hints(self) -> str:
        return (
            "The phrase 'common criticisms' suggests a threshold of at least 10% "
            "of the reviews that express criticism."
        )

    def filter_prompt_str(self) -> str | None:
        return "The {text} expresses a criticism or complaint"


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = ProportionalClaim1Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/yelp_restaurant_reviews/johns_roast_pork/summarize/proportional_claim_1_2026-02-10_18-31-26.json"
        ),
        dataset_key=("review_id",),
        text_field_name="text",
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
