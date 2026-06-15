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
from experiments.schemas import REVIEW_WITH_BUSINESS_SCHEMA


class ExistentialUniversalClaim1Evaluator(ClaimEvaluator):
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
                    "Identify whether the restaurant review {text} expresses a "
                    "negative sentiment towards the restaurant",
                    bool,
                ).alias("is_negative")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [bool_and(~col("is_negative")).alias("has_no_negative_reviews")],
                group_by=[col("business_id")],
            )
            .aggregate(
                [
                    bool_or(col("has_no_negative_reviews")).alias(
                        "some_locations_no_negative"
                    )
                ]
            )
            .check(col("some_locations_no_negative"))
        )

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("is_negative"),)


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = ExistentialUniversalClaim1Evaluator(
        name="existential_universal_claim_1",
        claim="Some McDonald's locations had no negative reviews.",
        hints="",
        schema=REVIEW_WITH_BUSINESS_SCHEMA,
        text_field_name="text",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/yelp_restaurant_reviews/mcdonalds_mo/compare_2026-02-10_19-11-51.json"
        ),
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
