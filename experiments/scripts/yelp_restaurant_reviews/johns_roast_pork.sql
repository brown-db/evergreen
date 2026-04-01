-- duckdb -c ".read experiments/scripts/yelp_restaurant_reviews/johns_roast_pork.sql"

.read experiments/scripts/yelp_restaurant_reviews/init.sql

-- Reviews for John's Roast Pork
copy (
    select *
    from yelp_restaurant_review
    where business_id = 'LM54ufrINJWoTN5imV8Etw'
    order by review_id
) to 'data/yelp_restaurant_reviews/johns_roast_pork.jsonl' (format json, array false);