from pathlib import Path

from evergreen.catalog.schema import Field, Schema
from experiments.compile_claim import compile_claim

if __name__ == "__main__":
    compile_claim(
        name="existential_claim_1",
        claim="Some reviewers enjoy the restaurant's chicken salad.",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/yelp_restaurant_reviews/johns_roast_pork/summarize_2026-02-10_18-24-08.json"
        ),
        schema=Schema(
            (Field("text", str, "The free-form text of a customer review."),),
            key=(),
        ),
    )
