-- duckdb -c ".read experiments/scripts/twitter_customer_support/uber.sql"

.read experiments/scripts/twitter_customer_support/init.sql

-- Customer support dialogs for Uber
copy (
    select *
    from twcs
    where company_id = 'Uber_Support'
    order by dialog_id
) to 'data/twitter_customer_support/uber.jsonl' (format json, array false);
