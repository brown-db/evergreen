from pathlib import Path

from evergreen.catalog.schema import Field, Schema
from experiments.compile_claim import compile_claim

if __name__ == "__main__":
    compile_claim(
        name="universal_cardinal_claim_1",
        claim="All McDonald's locations have multiple complaints about poor service "
        "quality.",
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
