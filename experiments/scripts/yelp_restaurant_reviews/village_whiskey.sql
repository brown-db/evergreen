-- duckdb -c ".read experiments/scripts/yelp_restaurant_reviews/village_whiskey.sql"

.read experiments/scripts/yelp_restaurant_reviews/init.sql

-- Reviews for Village Whiskey
copy (
    select *
    from yelp_restaurant_review
    where business_id = 'EtKSTHV5Qx_Q7Aur9o4kQQ'
    order by review_id
) to 'data/yelp_restaurant_reviews/village_whiskey.jsonl' (format json, array false);