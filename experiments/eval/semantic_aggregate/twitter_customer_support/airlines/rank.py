from pathlib import Path

from experiments.semantic_aggregate import SemanticAggregate

if __name__ == "__main__":
    semantic_aggregate = SemanticAggregate(
        name="rank",
        dataset_path=Path("data/twitter_customer_support/airlines.jsonl"),
        expr="'Company ID: ' || company_id || '\\nDialog: ' || dialog",
        prompt="Analyze the customer support dialogs for the different airline "
        "companies and describe the worst 3 airline companies regarding flight booking "
        "issues.",
        language_model="llama3.3-70b",
    )
    semantic_aggregate.execute()
