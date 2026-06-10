from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import Expr, bool_and, col, prompt
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import REVIEW_SCHEMA


class UniversalClaim1Evaluator(ClaimEvaluator):
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
                    "Identify whether the {text} mentions vegan options at the "
                    "restaurant",
                    bool,
                ).alias("mentions_vegan_options")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [bool_and(~col("mentions_vegan_options")).alias("none_mention_vegan")]
            )
            .check(col("none_mention_vegan"))
        )

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("mentions_vegan_options"),)


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = UniversalClaim1Evaluator(
        name="universal_claim_1",
        claim="The reviews do not mention any vegan options at the restaurant.",
        hints="",
        schema=REVIEW_SCHEMA,
        text_field_name="text",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/yelp_restaurant_reviews/johns_roast_pork/summarize_2026-02-10_18-24-08.json"
        ),
        random_seed=42,
    )
    claim_evaluator.evaluate(args)
