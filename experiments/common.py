import logging
import os
import tomllib
from pathlib import Path

DEFAULT_LANGUAGE_MODEL = "claude-opus-4-6"

EVALUATION_LANGUAGE_MODELS = (
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "qwen3-vl-235b-a22b",
    "qwen3-next-80b-a3b",
)

ENSEMBLE_LANGUAGE_MODELS = ("claude-opus-4-8", "openai-gpt-5.5", "gemini-3.5-flash")

MODEL_CONTEXT_WINDOW_TOKENS = {
    "claude-opus-4-6": 200_000,
    "claude-opus-4-5": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-sonnet-4-5": 200_000,
    "claude-haiku-4-5": 200_000,
}

# Input and output prices per million tokens for each language model
MODEL_PRICING = {
    # source: https://platform.claude.com/docs/en/about-claude/pricing
    "claude-opus-4-6": {
        "input": 5.0,
        "output": 25.0,
        "cache_write": 6.25,
        "cache_read": 0.50,
    },
    "claude-opus-4-5": {
        "input": 5.0,
        "output": 25.0,
        "cache_write": 6.25,
        "cache_read": 0.50,
    },
    "claude-sonnet-4-6": {
        "input": 3.0,
        "output": 15.0,
        "cache_write": 3.75,
        "cache_read": 0.30,
    },
    "claude-haiku-4-5": {
        "input": 1.0,
        "output": 5.0,
        "cache_write": 1.25,
        "cache_read": 0.10,
    },
    # source: https://www.alibabacloud.com/help/en/model-studio/model-pricing
    "qwen3-vl-235b-a22b": {"input": 0.287, "output": 1.147},
    "qwen3-next-80b-a3b": {"input": 0.144, "output": 0.574},
}

EMBEDDING_MODEL = "snowflake-arctic-embed-l-v2.0"

CONNECTION_NAME = "evergreen"

EXPERIMENTS_DIR = Path("experiments")

RESULTS_DIR = EXPERIMENTS_DIR / "results"

EXPERIMENT_DIR_ROOT_ENV_VAR = "EVERGREEN_EXPERIMENT_DIR_ROOT"

if EXPERIMENT_DIR_ROOT_ENV_VAR not in os.environ:
    raise ValueError(f"{EXPERIMENT_DIR_ROOT_ENV_VAR} environment variable is not set")

EXPERIMENT_DIR_ROOT = Path(os.environ[EXPERIMENT_DIR_ROOT_ENV_VAR])

LOGS_DIR = EXPERIMENT_DIR_ROOT / "logs"

CHECKPOINTS_DIR = EXPERIMENT_DIR_ROOT / "checkpoints"

TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(str(log_file), mode="w")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    for logger_name in ("evergreen", "experiments", "dspy"):
        parent_logger = logging.getLogger(logger_name)
        parent_logger.handlers.clear()
        parent_logger.setLevel(logging.INFO if logger_name == "dspy" else logging.DEBUG)
        parent_logger.addHandler(handler)


def load_snowflake_connection(connection_name: str) -> dict[str, str]:
    path = Path.home() / ".snowflake" / "connections.toml"
    with open(path, "rb") as f:
        config = tomllib.load(f)
    return config[connection_name]
