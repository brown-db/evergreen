from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import Expr, col, count_if, prompt, proportion
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import REVIEW_WITH_BUSINESS_SCHEMA


class ProportionalCardinalClaim1Evaluator(ClaimEvaluator):
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
                    "Identify whether the {text} mentions an incorrect order",
                    bool,
                ).alias("mentions_incorrect_order")
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
                    count_if(col("mentions_incorrect_order")).alias(
                        "incorrect_order_count"
                    )
                ],
                group_by=[col("business_id")],
            )
            .aggregate(
                [
                    proportion(col("incorrect_order_count") >= 2).alias(
                        "prop_with_multiple_incorrect"
                    )
                ]
            )
            .check(col("prop_with_multiple_incorrect") < 0.5)
        )

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("mentions_incorrect_order"),)


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = ProportionalCardinalClaim1Evaluator(
        name="proportional_cardinal_claim_1",
        claim="Only a minority of McDonald's locations had multiple reports of "
        "incorrect orders.",
        hints="",
        schema=REVIEW_WITH_BUSINESS_SCHEMA,
        text_field_name="text",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/yelp_restaurant_reviews/mcdonalds_mo/compare_2026-02-10_19-11-51.json"
        ),
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
