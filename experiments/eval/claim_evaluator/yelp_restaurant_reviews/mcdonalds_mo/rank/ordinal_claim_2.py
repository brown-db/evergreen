from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import Expr, col, prompt, proportion
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import REVIEW_WITH_BUSINESS_SCHEMA


class OrdinalClaim2Evaluator(ClaimEvaluator):
    def reference_query(
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
            .filter(prompt("The {text} mentions the service at the restaurant"))
            .map(
                prompt(
                    "Identify whether the {text} praises or speaks positively about "
                    "the service at the restaurant",
                    bool,
                ).alias("praises_service")
            )
            .log(
                str(
                    self._checkpoint_df_path(
                        CheckpointType.POST_SEM_OP, impl, language_models, trial_id
                    )
                )
            )
            .aggregate(
                [proportion(col("praises_service")).alias("service_praise_prop")],
                group_by=[col("business_id")],
            )
            .with_rank(col("service_praise_prop"))
            .filter(col("business_id").eq("9eYm5gwEOhBQdkg9ihV7EA"))
            .check(col("rank").eq(2))
        )

    def semantic_map_columns(self) -> tuple[Expr, ...]:
        return (col("praises_service"),)

    def filter_prompt_str(self) -> str | None:
        return "The {text} mentions the service at the restaurant"


if __name__ == "__main__":
    args = parse_claim_evaluator_args()
    claim_evaluator = OrdinalClaim2Evaluator(
        name="ordinal_claim_2",
        claim="The McDonald's location ranked #2 in terms of service has the "
        "Business ID 9eYm5gwEOhBQdkg9ihV7EA.",
        hints=(
            "Rank based on, among the reviews that mention the service, the "
            "proportion that speak positively about the service."
        ),
        schema=REVIEW_WITH_BUSINESS_SCHEMA,
        text_field_name="text",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/yelp_restaurant_reviews/mcdonalds_mo/rank_2026-02-20_20-01-21.json"
        ),
        random_seed=42,
        cache_id="ordinal_claims_1_and_2",
    )
    claim_evaluator.evaluate(args)
