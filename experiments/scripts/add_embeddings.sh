#!/bin/bash
set -euo pipefail

# Yelp Open Dataset
uv run python -m experiments.scripts.add_embeddings \
    --dataset_path data/yelp_restaurant_reviews/johns_roast_pork.jsonl \
    --fields text

uv run python -m experiments.scripts.add_embeddings \
    --dataset_path data/yelp_restaurant_reviews/mcdonalds_mo.jsonl \
    --fields text

uv run python -m experiments.scripts.add_embeddings \
    --dataset_path data/yelp_restaurant_reviews/village_whiskey.jsonl \
    --fields text

# Customer Support on Twitter Dataset
uv run python -m experiments.scripts.add_embeddings \
    --dataset_path data/twitter_customer_support/airlines.jsonl \
    --fields dialog

uv run python -m experiments.scripts.add_embeddings \
    --dataset_path data/twitter_customer_support/play_station.jsonl \
    --fields dialog

uv run python -m experiments.scripts.add_embeddings \
    --dataset_path data/twitter_customer_support/uber.jsonl \
    --fields dialog
