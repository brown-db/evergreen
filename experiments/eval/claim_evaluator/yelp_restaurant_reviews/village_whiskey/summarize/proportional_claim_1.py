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
from experiments.schemas import REVIEW_SCHEMA


class ProportionalClaim1Evaluator(ClaimEvaluator):
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
                    "Identify whether the {text} expresses a positive sentiment "
                    "towards the restaurant",
                    bool,
                ).alias("is_positive")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate([proportion(col("is_positive")).alias("positive_prop")])
            .check(col("positive_prop") > 0.5)
        )

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("is_positive"),)


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = ProportionalClaim1Evaluator(
        name="proportional_claim_1",
        claim="Village Whiskey has received majority positive reviews.",
        hints="",
        schema=REVIEW_SCHEMA,
        text_field_name="text",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/yelp_restaurant_reviews/village_whiskey/summarize_2026-02-10_17-32-55.json"
        ),
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
