from pathlib import Path

from experiments.semantic_aggregate import SemanticAggregate

if __name__ == "__main__":
    semantic_aggregate = SemanticAggregate(
        name="summarize",
        dataset_path=Path("data/twitter_customer_support/uber.jsonl"),
        expr="dialog",
        prompt="Summarize the customer issues and problematic agent responses in "
        "customer support dialogs for Uber. Integrate counts and statistics in your "
        "analysis where relevant.",
        language_model="llama3.3-70b",
    )
    semantic_aggregate.execute()
