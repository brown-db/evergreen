import os
from pathlib import Path

JSON_INDENT = 2

FIELD_NAME_REGEX = r"\{(\w+)\}"

EMBEDDING_FIELD_SUFFIX = "_embedding"
SENTENCE_EMBEDDINGS_FIELD_SUFFIX = "_sentence_embeddings"

CACHE_DIR_ROOT_ENV_VAR = "EVERGREEN_CACHE_DIR_ROOT"

if CACHE_DIR_ROOT_ENV_VAR not in os.environ:
    raise ValueError(f"{CACHE_DIR_ROOT_ENV_VAR} environment variable is not set")

CACHE_DIR_ROOT = Path(os.environ[CACHE_DIR_ROOT_ENV_VAR])
