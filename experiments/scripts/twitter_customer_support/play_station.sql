-- duckdb -c ".read experiments/scripts/twitter_customer_support/play_station.sql"

.read experiments/scripts/twitter_customer_support/init.sql

-- Customer support dialogs for PlayStation
copy (
    select *
    from twcs
    where company_id = 'AskPlayStation'
    order by dialog_id
) to 'data/twitter_customer_support/play_station.jsonl' (format json, array false);
