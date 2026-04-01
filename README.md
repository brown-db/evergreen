# Evergreen: Efficient Claim Verification for Semantic Aggregates

This repository contains the code and experiments for *Evergreen: Efficient Claim Verification for Semantic Aggregates*.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [DuckDB](https://duckdb.org/)
- [Snowflake account](https://signup.snowflake.com/) for [Cortex AI](https://www.snowflake.com/en/product/features/cortex/) language and embedding model access
- [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/)

## Setup

Install dependencies:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --all-groups
```

Configure [Snowflake connection](https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-connect#connecting-using-the-connections-toml-file) by creating `~/.snowflake/connections.toml`:
```toml
[evergreen]
account = "<account identifier>"
user = "<login name>"
password = "<programmatic access token>"
role = "<role>"
database = "<database>"
schema = "<schema>"
warehouse = "<warehouse>"
```

Set cache directory:
```sh
export EVERGREEN_CACHE_DIR_ROOT=~/.cache/evergreen/
```

Run tests (which depend on the docs) to ensure correct setup:
```sh
uv run mkdocs build
uv run pytest
```

## Data Preparation

Download the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) and extract to `data/yelp_dataset/`.
Extract evaluation datasets using DuckDB.

```sh
duckdb -c ".read experiments/scripts/yelp_restaurant_reviews/johns_roast_pork.sql"
duckdb -c ".read experiments/scripts/yelp_restaurant_reviews/mcdonalds_mo.sql"
duckdb -c ".read experiments/scripts/yelp_restaurant_reviews/village_whiskey.sql"
```

Add embeddings:
```sh
uv run python -m experiments.scripts.add_embeddings \
    --dataset_path data/yelp_restaurant_reviews/johns_roast_pork.jsonl \
    --fields text

uv run python -m experiments.scripts.add_embeddings \
    --dataset_path data/yelp_restaurant_reviews/mcdonalds_mo.jsonl \
    --fields text

uv run python -m experiments.scripts.add_embeddings \
    --dataset_path data/yelp_restaurant_reviews/village_whiskey.jsonl \
    --fields text
```

## Reproducing Results

Run the full evaluation:
```sh
uv run python -m experiments.scripts.run_claim_evaluators \
    --impls base_rm rag_agent \
    --lms claude-opus-4-6 claude-sonnet-4-6 claude-haiku-4-5

uv run python -m experiments.scripts.run_claim_evaluators \
    --impls evg_ref

uv run python -m experiments.scripts.run_claim_evaluators \
    --impls evg_opt \
    --lms claude-opus-4-6 claude-sonnet-4-6 claude-haiku-4-5 llama4-maverick llama4-scout llama3.1-8b

uv run python -m experiments.scripts.run_claim_evaluators \
    --impls evg_unopt \
    --lms claude-opus-4-6

uv run python -m experiments.scripts.run_claim_evaluators \
    --impls evg_abl_no_es evg_abl_no_rs evg_abl_no_est evg_abl_no_fus evg_abl_no_sf evg_abl_no_cache \
    --lms claude-haiku-4-5

uv run python -m experiments.scripts.run_claim_evaluators --eval_sim_filter
```

Consider moving any existing results in `experiments/results/` to a separate directory to avoid overwriting or double counting results.

Generate figures:
```sh
uv run python -m experiments.scripts.plot_results
```

Figures are saved to `experiments/figures/`.
