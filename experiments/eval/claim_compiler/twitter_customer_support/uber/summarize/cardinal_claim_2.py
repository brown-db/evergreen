from pathlib import Path

from evergreen.catalog.schema import Field, Schema
from experiments.compile_claim import compile_claim

if __name__ == "__main__":
    compile_claim(
        name="cardinal_claim_2",
        claim="Less than 750 customers complained about poor driver behavior.",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/twitter_customer_support/uber/summarize_2026-06-01_14-04-38.json"
        ),
        schema=Schema(
            (Field("dialog", str, "The customer support dialog."),),
            key=(),
        ),
    )
