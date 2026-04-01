from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import Expr, bool_or, col, prompt
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)


class ExistentialClaim1Evaluator(ClaimEvaluator):
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
                    "The {text} expresses enjoyment of the restaurant's chicken salad",
                    bool,
                ).alias("enjoys_chicken_salad")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [bool_or(col("enjoys_chicken_salad")).alias("some_enjoy_chicken_salad")]
            )
            .check(col("some_enjoy_chicken_salad"))
        )

    def check_predicate(self) -> Expr:
        return col("some_enjoy_chicken_salad")

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("enjoys_chicken_salad"),)

    def hints(self) -> str:
        return "'Some' suggests a threshold of at least one."


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = ExistentialClaim1Evaluator(
        claim_compilation_result_path=Path(
            "experiments/results/claim_compiler/yelp_restaurant_reviews/johns_roast_pork/summarize/existential_claim_1_2026-02-15_16-39-02.json"
        ),
        dataset_key=("review_id",),
        text_field_name="text",
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
