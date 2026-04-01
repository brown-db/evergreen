from pathlib import Path

from experiments.semantic_aggregate import SemanticAggregate

if __name__ == "__main__":
    semantic_aggregate = SemanticAggregate(
        name="compare",
        dataset_path=Path("data/yelp_restaurant_reviews/mcdonalds_mo.jsonl"),
        expr="'Business ID: ' || business_id || '\\nReview: ' || text",
        prompt="Compare the reviews for the different McDonald''s locations. "
        "Highlight the commonalities and differences between the restaurant "
        "locations.",
        language_model="llama3.3-70b",
    )
    semantic_aggregate.execute()
