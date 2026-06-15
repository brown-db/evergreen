from pathlib import Path

from evergreen.data_frame import DataFrame
from evergreen.planner.logical.expr import col, prompt, proportion
from experiments.claim_evaluator import (
    CheckpointType,
    ClaimEvaluator,
    Implementation,
    parse_claim_evaluator_args,
)
from experiments.schemas import REVIEW_WITH_BUSINESS_SCHEMA


class OrdinalClaim4Evaluator(ClaimEvaluator):
    NAME = "ordinal_claim_4"
    CLAIM = (
        "The McDonald's location ranked #2 in terms of service has the "
        "Business ID XlbFKW_Keun8qh1S5m3QgQ."
    )
    HINTS = (
        "Rank based on, among the reviews that mention the service, the "
        "proportion that speak positively about the service."
    )
    SCHEMA = REVIEW_WITH_BUSINESS_SCHEMA
    TEXT_FIELD_NAME = "text"
    AGG_RESULT_PATH = Path(
        "experiments/results/semantic_aggregate/yelp_restaurant_reviews/mcdonalds_mo/rank_2026-02-20_20-01-21.json"
    )
    CACHE_ID = "ordinal_claims_3_and_4"

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
            .filter(
                prompt(
                    "The restaurant review {text} describes the restaurant's service"
                )
            )
            .map(
                prompt(
                    "Identify whether the restaurant review {text} praises the "
                    "restaurant's service",
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
            .filter(col("business_id").eq("XlbFKW_Keun8qh1S5m3QgQ"))
            .check(col("rank").eq(2))
        )


if __name__ == "__main__":
    OrdinalClaim4Evaluator().evaluate(parse_claim_evaluator_args())
