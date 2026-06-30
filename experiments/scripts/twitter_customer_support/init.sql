create or replace table twcs as 
select *
from read_json_auto('data/twitter_customer_support/twcs.jsonl');
