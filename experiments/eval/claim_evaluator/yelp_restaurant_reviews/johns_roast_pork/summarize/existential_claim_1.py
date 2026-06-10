from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import Expr, bool_or, col, prompt
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import REVIEW_SCHEMA


class ExistentialClaim1Evaluator(ClaimEvaluator):
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
                    "Identify whether the {text} expresses enjoyment of the "
                    "restaurant's chicken salad",
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

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("enjoys_chicken_salad"),)


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = ExistentialClaim1Evaluator(
        name="existential_claim_1",
        claim="Some reviewers enjoy the restaurant's chicken salad.",
        hints="",
        schema=REVIEW_SCHEMA,
        text_field_name="text",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/yelp_restaurant_reviews/johns_roast_pork/summarize_2026-02-10_18-24-08.json"
        ),
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
