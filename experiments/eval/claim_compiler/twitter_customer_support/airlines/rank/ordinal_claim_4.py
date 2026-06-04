from pathlib import Path

from evergreen.catalog.schema import Field, Schema
from experiments.compile_claim import compile_claim

if __name__ == "__main__":
    compile_claim(
        name="ordinal_claim_4",
        claim="AirAsiaSupport has the second most customer support dialogs with "
        "complaints about flight booking issues.",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/twitter_customer_support/airlines/rank_2026-06-03_18-22-36.json"
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
