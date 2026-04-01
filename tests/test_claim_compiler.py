from evergreen.catalog.schema import Field, Schema
from evergreen.claim_compiler import ClaimCompiler
from evergreen.model.config import CortexModelConfig
from tests.conftest import DEFAULT_CONNECTION_NAME, DEFAULT_LANGUAGE_MODEL


def test_claim_compiler():
    model_config = CortexModelConfig(
        language_models=[DEFAULT_LANGUAGE_MODEL],
        embedding_model="",
        connection_name=DEFAULT_CONNECTION_NAME,
    )
    language_model = model_config.create_language_model(cache_dir=None)
    compiler = ClaimCompiler(language_model)

    agg_prompt = "Summarize the reviews for this restaurant."
    schema = Schema(
        (
            Field("id", int, "The ID of the review"),
            Field("review", str, "The text of the review"),
        ),
        ("id",),
    )
    claim = "Some customers wait 20+ minutes to be acknowledged."
    result = compiler.compile(agg_prompt, schema, claim)

    assert len(result.response) > 0
    assert "check(" in result.query
    assert len(result.language_model_metrics) == 1
    assert result.language_model_metrics[0].prompt_count == 1
    assert result.latency > 0
