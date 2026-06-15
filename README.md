# Evergreen: Efficient Claim Verification for Semantic Aggregates

This repository contains the code and experiments for [*Evergreen: Efficient Claim Verification for Semantic Aggregates*](https://arxiv.org/abs/2604.26180).

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [DuckDB](https://duckdb.org/)
- [Snowflake account](https://signup.snowflake.com/) for [Cortex AI](https://www.snowflake.com/en/product/features/cortex/) language and embedding model access
- [Deno](https://deno.com/) (only required for [RLM](https://dspy.ai/api/modules/RLM/) experiments)

## Setup

Install dependencies:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

To run experiments with [RLM](https://dspy.ai/api/modules/RLM/), install [Deno](https://deno.com/) and then restart your shell (or re-source your shell profile) so the updated `PATH` takes effect:
```sh
curl -fsSL https://deno.land/install.sh | sh
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

Set log directory (holds experiment logs and dataframe checkpoints):
```sh
export EVERGREEN_EXPERIMENT_DIR_ROOT=experiments/
```

Build the documentation, which generates `site/llms-full.txt` (the API reference used by the claim compiler):
```sh
uv run mkdocs build
```

Run tests to ensure correct setup:
```sh
uv run pytest
```

## Data Preparation

The datasets are sourced from the [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) and the [Customer Support on Twitter Dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).

Download both datasets and build the evaluation datasets:
```sh
./experiments/scripts/prepare_data.sh
```

Add embeddings (requires a configured Snowflake connection):
```sh
./experiments/scripts/add_embeddings.sh
```

## Reproducing Results

Run the full evaluation:
```sh
uv run python -m experiments.scripts.run_claim_evaluators \
    --impls base_rm rag_agent rlm \
    --lms claude-opus-4-6 claude-sonnet-4-6 claude-haiku-4-5

uv run python -m experiments.scripts.run_claim_evaluators \
    --impls evg_ref

uv run mkdocs build && uv run python -m experiments.scripts.run_claim_evaluators --compile

uv run python -m experiments.scripts.run_claim_evaluators \
    --impls evg_opt evg_opt_ref_query \
    --lms claude-opus-4-6 claude-sonnet-4-6 claude-haiku-4-5 llama4-maverick llama4-scout llama3.1-8b

uv run python -m experiments.scripts.run_claim_evaluators \
    --impls evg_unopt evg_unopt_ref_query \
    --lms claude-opus-4-6 claude-sonnet-4-6 claude-haiku-4-5

uv run python -m experiments.scripts.run_claim_evaluators \
    --impls evg_abl_no_es evg_abl_no_rs evg_abl_no_ecs evg_abl_no_of evg_abl_no_sf evg_abl_no_pc \
    --lms claude-haiku-4-5

uv run python -m experiments.scripts.run_claim_evaluators --eval_sim_filter
```

Consider moving any existing results in `experiments/results/` to a separate directory to avoid overwriting or double counting results.

Generate figures:
```sh
uv run python -m experiments.scripts.plot_results
```

Figures are saved to `experiments/figures/`.
