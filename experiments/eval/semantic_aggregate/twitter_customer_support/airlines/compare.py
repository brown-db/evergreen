from pathlib import Path

from experiments.semantic_aggregate import SemanticAggregate

if __name__ == "__main__":
    semantic_aggregate = SemanticAggregate(
        name="compare",
        dataset_path=Path("data/twitter_customer_support/airlines.jsonl"),
        expr="'Company ID: ' || company_id || '\\nDialog: ' || dialog",
        prompt="Compare the customer support dialogs for the different airline "
        "companies. Highlight the commonalities and differences between the companies "
        "and integrate counts and statistics in your analysis where relevant.",
        language_model="llama3.3-70b",
    )
    semantic_aggregate.execute()
