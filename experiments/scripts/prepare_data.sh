#!/bin/bash
set -euo pipefail

# Yelp Open Dataset
mkdir -p data/yelp_dataset
curl -L -o data/yelp_dataset/Yelp-JSON.zip \
  https://business.yelp.com/external-assets/files/Yelp-JSON.zip
unzip data/yelp_dataset/Yelp-JSON.zip -d data/yelp_dataset/
tar -xf "data/yelp_dataset/Yelp JSON/yelp_dataset.tar" -C data/yelp_dataset/

duckdb -c ".read experiments/scripts/yelp_restaurant_reviews/johns_roast_pork.sql"
duckdb -c ".read experiments/scripts/yelp_restaurant_reviews/mcdonalds_mo.sql"
duckdb -c ".read experiments/scripts/yelp_restaurant_reviews/village_whiskey.sql"

# Customer Support on Twitter Dataset
mkdir -p data/twitter_customer_support
curl -L -o data/twitter_customer_support/customer-support-on-twitter.zip \
  https://www.kaggle.com/api/v1/datasets/download/thoughtvector/customer-support-on-twitter
unzip data/twitter_customer_support/customer-support-on-twitter.zip \
  -d data/twitter_customer_support/

uv run python -m experiments.scripts.twitter_customer_support.reconstruct_dialogs \
  --dataset_path data/twitter_customer_support/twcs.csv \
  --output_dataset_path data/twitter_customer_support/twcs.jsonl

duckdb -c ".read experiments/scripts/twitter_customer_support/airlines.sql"
duckdb -c ".read experiments/scripts/twitter_customer_support/play_station.sql"
duckdb -c ".read experiments/scripts/twitter_customer_support/uber.sql"
