create or replace table yelp_business as 
select *
from read_json_auto('data/yelp_dataset/yelp_academic_dataset_business.json');

create or replace table yelp_review as
select *
from read_json_auto('data/yelp_dataset/yelp_academic_dataset_review.json');

create or replace table yelp_restaurant_business as
select *
from yelp_business
where contains(categories, 'Restaurants');

create or replace table yelp_restaurant_review as
select review_id,
        user_id,
        yelp_review.business_id,
        yelp_restaurant_business.name,
        yelp_restaurant_business.address,
        yelp_restaurant_business.city,
        yelp_restaurant_business.state,
        yelp_restaurant_business.postal_code,
        yelp_restaurant_business.categories,
        yelp_review.stars,
        date,
        text
from yelp_review
inner join yelp_restaurant_business
on yelp_review.business_id = yelp_restaurant_business.business_id;