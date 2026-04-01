from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import Expr, col, count_if, prompt
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
                    "Identify whether the {text} complains about the restaurant's "
                    "service quality",
                    bool,
                ).alias("complains_about_service")
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
                    count_if(col("complains_about_service")).alias(
                        "service_complaint_count"
                    )
                ]
            )
            .check(col("service_complaint_count") < 5)
        )

    def check_predicate(self) -> Expr:
        return col("service_complaint_count") < 5

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("complains_about_service"),)

    def hints(self) -> str:
        return "The phrase 'less than a handful' suggests a threshold of less than 5."


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = CardinalClaim1Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/yelp_restaurant_reviews/johns_roast_pork/summarize/cardinal_claim_1_2026-02-17_19-39-26.json"
        ),
        dataset_key=("review_id",),
        text_field_name="text",
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
