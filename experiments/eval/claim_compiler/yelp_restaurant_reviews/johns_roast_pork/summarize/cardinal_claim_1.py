from pathlib import Path

from evergreen.catalog.schema import Field, Schema
from experiments.compile_claim import compile_claim

if __name__ == "__main__":
    compile_claim(
        name="cardinal_claim_1",
        claim="Less than a handful of customers complained about the restaurant's "
        "service quality.",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/yelp_restaurant_reviews/johns_roast_pork/summarize_2026-02-10_18-24-08.json"
        ),
        schema=Schema(
            (Field("text", str, "The free-form text of a customer review."),),
            key=(),
        ),
    )
