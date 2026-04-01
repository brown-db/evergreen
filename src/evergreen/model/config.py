from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from snowflake.snowpark import Session

from evergreen.model.embedding_model import CortexEmbeddingModel, EmbeddingModel
from evergreen.model.language_model import (
    CortexLanguageModel,
    EnsembleLanguageModel,
    LanguageModel,
)


class ModelConfig(ABC):
    @abstractmethod
    def create_language_model(self, cache_dir: Path | None) -> LanguageModel:
        pass

    @abstractmethod
    def create_embedding_model(self, cache_dir: Path | None) -> EmbeddingModel:
        pass

    @abstractmethod
    def create_optimizer_language_model(self, cache_dir: Path | None) -> LanguageModel:
        pass


class CortexModelConfig(ModelConfig):
    def __init__(
        self,
        language_models: Sequence[str],
        embedding_model: str,
        connection_name: str,
    ) -> None:
        if len(language_models) == 0:
            raise ValueError("At least one language model must be provided")

        self._language_models = language_models
        self._embedding_model = embedding_model
        self._connection_name = connection_name
        self._session: Session | None = None

    def _get_or_create_session(self) -> Session:
        if self._session is None:
            self._session = Session.builder.config(
                "connection_name", self._connection_name
            ).create()
        return self._session

    def create_language_model(self, cache_dir: Path | None) -> LanguageModel:
        session = self._get_or_create_session()
        if len(self._language_models) == 1:
            return CortexLanguageModel(self._language_models[0], cache_dir, session)
        return EnsembleLanguageModel(
            tuple(
                CortexLanguageModel(model, cache_dir, session)
                for model in self._language_models
            ),
            cache_dir,
        )

    def create_embedding_model(self, cache_dir: Path | None) -> EmbeddingModel:
        session = self._get_or_create_session()
        return CortexEmbeddingModel(self._embedding_model, cache_dir, session)

    def create_optimizer_language_model(self, cache_dir: Path | None) -> LanguageModel:
        session = self._get_or_create_session()
        return CortexLanguageModel("claude-opus-4-6", cache_dir, session)
