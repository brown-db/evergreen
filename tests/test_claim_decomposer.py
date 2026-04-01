from evergreen.claim_decomposer import ClaimDecomposer
from evergreen.model.config import CortexModelConfig
from tests.conftest import DEFAULT_CONNECTION_NAME, DEFAULT_LANGUAGE_MODEL


def test_claim_decomposer():
    model_config = CortexModelConfig(
        language_models=[DEFAULT_LANGUAGE_MODEL],
        embedding_model="",
        connection_name=DEFAULT_CONNECTION_NAME,
    )
    language_model = model_config.create_language_model(cache_dir=None)
    decomposer = ClaimDecomposer(language_model)

    text = (
        "Flowing Tide Pub emerges as a popular local Reno chain offering solid pub "
        "fare in a casual, nautical-themed sports bar atmosphere. The establishment is "
        "particularly known for its exceptional dual happy hours (2-6pm and "
        "10pm-midnight) featuring 50% off drinks and $6 appetizers, which many "
        'consider "the best happy hour in Reno."'
    )

    claims = decomposer.decompose(text)

    assert len(claims) > 0
    claims_text = " ".join(claims).lower()
    assert "flowing tide pub" in claims_text
    assert "happy hour" in claims_text
