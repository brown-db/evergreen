import pytest

from evergreen.common.utils import clear_all_caches
from evergreen.core.session_context import SessionContext
from evergreen.model.config import CortexModelConfig

DEFAULT_LANGUAGE_MODEL = "claude-sonnet-4-6"
DEFAULT_EMBEDDING_MODEL = "snowflake-arctic-embed-l-v2.0"
DEFAULT_CONNECTION_NAME = "evergreen"


@pytest.fixture
def default_ctx() -> SessionContext:
    return SessionContext()


@pytest.fixture
def optimized_ctx() -> SessionContext:
    ctx = SessionContext.create_optimized()
    ctx.register_model_config(
        CortexModelConfig(
            [DEFAULT_LANGUAGE_MODEL], DEFAULT_EMBEDDING_MODEL, DEFAULT_CONNECTION_NAME
        )
    )
    return ctx


@pytest.fixture
def ctx_with_model() -> SessionContext:
    ctx = SessionContext()
    ctx.register_model_config(
        CortexModelConfig(
            [DEFAULT_LANGUAGE_MODEL], DEFAULT_EMBEDDING_MODEL, DEFAULT_CONNECTION_NAME
        )
    )
    return ctx


@pytest.fixture
def ctx_with_early_stop() -> SessionContext:
    ctx = SessionContext()
    ctx.enable_early_stop()
    return ctx


@pytest.fixture
def ctx_with_fusion() -> SessionContext:
    ctx = SessionContext()
    ctx.enable_fusion()
    ctx.enable_early_stop()
    ctx.enable_minimal_provenance()
    ctx.register_model_config(
        CortexModelConfig(
            [DEFAULT_LANGUAGE_MODEL], DEFAULT_EMBEDDING_MODEL, DEFAULT_CONNECTION_NAME
        )
    )
    return ctx


@pytest.fixture
def ctx_with_estimation() -> SessionContext:
    ctx = SessionContext()
    ctx.enable_estimation()
    return ctx


@pytest.fixture
def ctx_with_relevance_sort() -> SessionContext:
    ctx = SessionContext()
    ctx.enable_relevance_sort()
    ctx.register_model_config(
        CortexModelConfig(
            [DEFAULT_LANGUAGE_MODEL], DEFAULT_EMBEDDING_MODEL, DEFAULT_CONNECTION_NAME
        )
    )
    return ctx


@pytest.fixture
def ctx_with_similarity_filter() -> SessionContext:
    ctx = SessionContext()
    ctx.enable_similarity_filter()
    ctx.register_model_config(
        CortexModelConfig(
            [DEFAULT_LANGUAGE_MODEL], DEFAULT_EMBEDDING_MODEL, DEFAULT_CONNECTION_NAME
        )
    )
    return ctx


@pytest.fixture
def ctx_with_caching() -> SessionContext:
    ctx = SessionContext()
    ctx.enable_cache()
    ctx.register_model_config(
        CortexModelConfig(
            [DEFAULT_LANGUAGE_MODEL], DEFAULT_EMBEDDING_MODEL, DEFAULT_CONNECTION_NAME
        )
    )
    return ctx


@pytest.fixture(autouse=True, scope="session")
def clear_all_caches_fixture():
    """Clear all caches after all tests complete."""
    yield
    clear_all_caches()
