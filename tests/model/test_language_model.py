from enum import Enum

import pytest

from evergreen.model.config import CortexModelConfig
from tests.conftest import (
    DEFAULT_CONNECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LANGUAGE_MODEL,
)


class TestCortexLanguageModel:
    @pytest.mark.parametrize(
        "language_models",
        [
            [DEFAULT_LANGUAGE_MODEL],
            [
                "claude-opus-4-6",
                "openai-gpt-5.5",
                "gemini-3.5-flash",
            ],
        ],
    )
    def test_prompt(self, language_models: list[str]):
        model_config = CortexModelConfig(
            language_models, DEFAULT_EMBEDDING_MODEL, DEFAULT_CONNECTION_NAME
        )
        model = model_config.create_language_model(cache_dir=None)

        responses = model.prompt(["Is 2 + 2 = 4? Output true or false."], bool)
        assert responses[0] is True
        assert len(model.metrics()) == len(language_models)
        assert all(metrics.prompt_count == 1 for metrics in model.metrics())

        responses = model.prompt(["Is 2 + 2 = 5? Output true or false."], bool)
        assert responses[0] is False
        assert all(metrics.prompt_count == 2 for metrics in model.metrics())

        class Sentiment(Enum):
            POSITIVE = "positive"
            NEGATIVE = "negative"
            NEUTRAL = "neutral"

        responses = model.prompt(
            ["Is the sentence positive, negative, or neutral? 'I love this product.'"],
            Sentiment,
        )
        assert responses[0] == Sentiment.POSITIVE
        assert all(metrics.prompt_count == 3 for metrics in model.metrics())

    def test_batch_prompt(self):
        model_config = CortexModelConfig(
            [DEFAULT_LANGUAGE_MODEL], DEFAULT_EMBEDDING_MODEL, DEFAULT_CONNECTION_NAME
        )
        model = model_config.create_language_model(cache_dir=None)

        responses = model.prompt(
            [
                "Is 2 + 2 = 3? Output true or false.",
                "Is 2 + 2 = 4? Output true or false.",
                "Is 2 + 2 = 5? Output true or false.",
            ],
            bool,
        )
        assert responses == [False, True, False]
        assert all(metrics.prompt_count == 3 for metrics in model.metrics())
