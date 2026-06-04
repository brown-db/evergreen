from pathlib import Path

from evergreen.catalog.schema import Field, Schema
from experiments.compile_claim import compile_claim

if __name__ == "__main__":
    compile_claim(
        name="cardinal_proportional_claim_1",
        claim="There are 4 airline companies where over 10% of customer complaints are "
        "regarding flight delays.",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/twitter_customer_support/airlines/compare_2026-06-01_14-04-34.json"
        ),
        schema=Schema(
            (
                Field(
                    "company_id", str, "The unique identifier of the airline company."
                ),
                Field("dialog", str, "The customer support dialog."),
            ),
            key=(),
        ),
    )
