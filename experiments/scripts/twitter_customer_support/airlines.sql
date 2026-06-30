-- duckdb -c ".read experiments/scripts/twitter_customer_support/airlines.sql"

.read experiments/scripts/twitter_customer_support/init.sql

-- Customer support dialogs for airlines
copy (
    select *
    from twcs
    where company_id in ('VirginAmerica', 'AlaskaAir', 'VirginAtlantic', 'JetBlue', 'AirAsiaSupport', 'SouthwestAir')
    order by company_id, dialog_id
) to 'data/twitter_customer_support/airlines.jsonl' (format json, array false);
