-- duckdb -c ".read experiments/scripts/yelp_restaurant_reviews/mcdonalds_mo.sql"

.read experiments/scripts/yelp_restaurant_reviews/init.sql

-- Reviews for all McDonald's in Missouri (MO)
copy (
    select *
    from yelp_restaurant_review
    where contains(name, 'McDonald''s') and state = 'MO'
    order by business_id, review_id
) to 'data/yelp_restaurant_reviews/mcdonalds_mo.jsonl' (format json, array false);