from pathlib import Path

from evergreen.catalog.schema import Field, Schema
from experiments.compile_claim import compile_claim

if __name__ == "__main__":
    compile_claim(
        name="ordinal_claim_3",
        claim="The top-ranked McDonald's location in terms of service has the "
        "Business ID 9eYm5gwEOhBQdkg9ihV7EA.",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/yelp_restaurant_reviews/mcdonalds_mo/rank_2026-02-20_20-01-21.json"
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
