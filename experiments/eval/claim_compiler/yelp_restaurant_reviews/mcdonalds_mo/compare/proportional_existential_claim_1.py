from pathlib import Path

from evergreen.catalog.schema import Field, Schema
from experiments.compile_claim import compile_claim

if __name__ == "__main__":
    compile_claim(
        name="proportional_existential_claim_1",
        claim="The majority of McDonald's locations had reports of cold food.",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/yelp_restaurant_reviews/mcdonalds_mo/compare_2026-02-10_19-11-51.json"
        ),
        schema=Schema(
            (
                Field(
                    "business_id",
                    str,
                    "The unique identifier of the reviewed restaurant.",
                ),
                Field("text", str, "The free-form text of a customer review."),
            ),
            key=(),
        ),
    )
