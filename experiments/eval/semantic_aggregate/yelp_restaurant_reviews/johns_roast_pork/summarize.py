from pathlib import Path

from experiments.semantic_aggregate import SemanticAggregate

if __name__ == "__main__":
    semantic_aggregate = SemanticAggregate(
        name="summarize",
        dataset_path=Path("data/yelp_restaurant_reviews/johns_roast_pork.jsonl"),
        expr="text",
        prompt="Summarize what customers are saying about John''s Roast Pork.",
        language_model="llama3.3-70b",
    )
    semantic_aggregate.execute()
