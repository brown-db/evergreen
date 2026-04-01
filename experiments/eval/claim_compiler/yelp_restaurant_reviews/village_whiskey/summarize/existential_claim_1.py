from pathlib import Path

from evergreen.catalog.schema import Field, Schema
from experiments.compile_claim import compile_claim

if __name__ == "__main__":
    compile_claim(
        name="existential_claim_1",
        claim="Some reviewers enjoyed the restaurant's bacon fries.",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/yelp_restaurant_reviews/village_whiskey/summarize_2026-02-10_17-32-55.json"
        ),
        schema=Schema(
            (Field("text", str, "The free-form text of a customer review."),),
            key=(),
        ),
    )
