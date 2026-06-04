from pathlib import Path

from evergreen.catalog.schema import Field, Schema
from experiments.compile_claim import compile_claim

if __name__ == "__main__":
    compile_claim(
        name="cardinal_claim_1",
        claim="240 customers are experiencing problems with their accounts or login "
        "issues.",
        agg_result_path=Path(
            "experiments/results/semantic_aggregate/twitter_customer_support/play_station/summarize_2026-06-01_14-04-42.json"
        ),
        schema=Schema(
            (Field("dialog", str, "The customer support dialog."),),
            key=(),
        ),
    )
