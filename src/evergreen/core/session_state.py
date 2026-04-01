import uuid
from pathlib import Path

from evergreen.common.constants import CACHE_DIR_ROOT
from evergreen.model.config import ModelConfig
from evergreen.model.embedding_model import EmbeddingModel, EmbeddingModelMetrics
from evergreen.model.language_model import LanguageModel, LanguageModelMetrics
from evergreen.planner.logical.optimizer import LogicalOptimizer
from evergreen.planner.logical.plan import LogicalPlan
from evergreen.planner.physical.plan import PhysicalPlan
from evergreen.planner.physical.planner import PhysicalPlanner


class SessionState:
    def __init__(self) -> None:
        self._session_id = uuid.uuid4().hex
        self._logical_optimizer = LogicalOptimizer()
        self._physical_planner = PhysicalPlanner()
        self._language_model: LanguageModel | None = None
        self._embedding_model: EmbeddingModel | None = None
        self._optimizer_language_model: LanguageModel | None = None
        self._batch_size: int = 1
        self._early_stop_enabled: bool = False
        self._fusion_enabled: bool = False
        self._minimal_provenance: bool = False
        self._relevance_sort_enabled: bool = False
        self._similarity_threshold: float | None = None
        self._confidence_level: float | None = None
        self._relative_error: float = 0.0
        self._random_seed: int | None = None
        self._cache_dir: Path | None = None

    def language_model(self) -> LanguageModel:
        if self._language_model is None:
            raise ValueError("Language model not registered")
        return self._language_model

    def embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            raise ValueError("Embedding model not registered")
        return self._embedding_model

    def optimizer_language_model(self) -> LanguageModel:
        if self._optimizer_language_model is None:
            raise ValueError("Optimizer language model not registered")
        return self._optimizer_language_model

    def language_model_metrics(self) -> tuple[LanguageModelMetrics, ...]:
        if self._language_model is not None:
            return self._language_model.metrics()
        return ()

    def embedding_model_metrics(self) -> tuple[EmbeddingModelMetrics, ...]:
        if self._embedding_model is not None:
            return self._embedding_model.metrics()
        return ()

    def optimizer_language_model_metrics(self) -> tuple[LanguageModelMetrics, ...]:
        if self._optimizer_language_model is not None:
            return self._optimizer_language_model.metrics()
        return ()

    def batch_size(self) -> int:
        return self._batch_size

    def early_stop_enabled(self) -> bool:
        return self._early_stop_enabled

    def fusion_enabled(self) -> bool:
        return self._fusion_enabled

    def minimal_provenance(self) -> bool:
        return self._minimal_provenance

    def relevance_sort_enabled(self) -> bool:
        return self._relevance_sort_enabled

    def similarity_threshold(self) -> float | None:
        return self._similarity_threshold

    def confidence_level(self) -> float | None:
        return self._confidence_level

    def relative_error(self) -> float:
        return self._relative_error

    def random_seed(self) -> int | None:
        return self._random_seed

    def register_model_config(self, model_config: ModelConfig) -> None:
        self._language_model = model_config.create_language_model(self._cache_dir)
        self._embedding_model = model_config.create_embedding_model(self._cache_dir)
        self._optimizer_language_model = model_config.create_optimizer_language_model(
            self._cache_dir
        )

    def enable_batching(self, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        self._batch_size = batch_size

    def enable_early_stop(self) -> None:
        self._early_stop_enabled = True

    def enable_fusion(self) -> None:
        self._fusion_enabled = True

    def enable_minimal_provenance(self) -> None:
        self._minimal_provenance = True

    def enable_relevance_sort(self) -> None:
        self._relevance_sort_enabled = True
        self._early_stop_enabled = True

    def enable_similarity_filter(self, similarity_threshold: float) -> None:
        if similarity_threshold <= -1.0 or similarity_threshold >= 1.0:
            raise ValueError("Similarity threshold must be in (-1, 1)")

        self._similarity_threshold = similarity_threshold

    def enable_estimation(
        self,
        confidence_level: float,
        relative_error: float,
        random_seed: int | None,
    ) -> None:
        if confidence_level <= 0.0 or confidence_level >= 1.0:
            raise ValueError("Confidence level must be in (0, 1)")

        if relative_error <= 0.0 or relative_error >= 1.0:
            raise ValueError("Relative error must be in (0, 1)")

        self._confidence_level = confidence_level
        self._relative_error = relative_error
        self._random_seed = random_seed
        self._early_stop_enabled = True

    def enable_cache(self, cache_id: str | None) -> None:
        if cache_id is None:
            cache_id = self._session_id
        self._cache_dir = CACHE_DIR_ROOT / cache_id

    def clear_language_model_metrics(self) -> None:
        if self._language_model is not None:
            self._language_model.clear_metrics()

    def clear_embedding_model_metrics(self) -> None:
        if self._embedding_model is not None:
            self._embedding_model.clear_metrics()

    def clear_optimizer_language_model_metrics(self) -> None:
        if self._optimizer_language_model is not None:
            self._optimizer_language_model.clear_metrics()

    def optimize_logical_plan(self, logical_plan: LogicalPlan) -> LogicalPlan:
        return self._logical_optimizer.optimize(logical_plan, self)

    def create_physical_plan(self, logical_plan: LogicalPlan) -> PhysicalPlan:
        return self._physical_planner.create_physical_plan(logical_plan, self)
