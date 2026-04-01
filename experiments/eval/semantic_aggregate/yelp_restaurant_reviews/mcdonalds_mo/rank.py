from pathlib import Path

from experiments.semantic_aggregate import SemanticAggregate

if __name__ == "__main__":
    semantic_aggregate = SemanticAggregate(
        name="rank",
        dataset_path=Path("data/yelp_restaurant_reviews/mcdonalds_mo.jsonl"),
        expr="'Business ID: ' || business_id || '\\nReview: ' || text",
        prompt="Analyze the reviews for the the different McDonald''s locations "
        "and describe the top 3 locations in terms of their service.",
        language_model="llama3.3-70b",
    )
    semantic_aggregate.execute()
