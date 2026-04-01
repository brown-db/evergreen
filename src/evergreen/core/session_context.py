from __future__ import annotations

import json
import pickle
from typing import cast

from evergreen.catalog.schema import Field, Schema
from evergreen.catalog.table import InMemoryTable, JsonlTable, TableProvider
from evergreen.core.session_state import SessionState
from evergreen.data_frame import DataFrame
from evergreen.model.config import ModelConfig
from evergreen.planner.logical.plan_builder import LogicalPlanBuilder
from evergreen.storage.row import Row

DEFAULT_BATCH_SIZE: int = 32
DEFAULT_SIMILARITY_THRESHOLD: float = 0.15
DEFAULT_CONFIDENCE_LEVEL: float = 0.95
DEFAULT_RELATIVE_ERROR: float = 0.05


class SessionContext:
    def __init__(self) -> None:
        self._session_state = SessionState()

    def session_state(self) -> SessionState:
        return self._session_state

    @classmethod
    def create_optimized(
        cls, random_seed: int | None = None, cache_id: str | None = None
    ) -> SessionContext:
        ctx = SessionContext()
        ctx.enable_batching()
        ctx.enable_early_stop()
        ctx.enable_fusion()
        ctx.enable_minimal_provenance()
        ctx.enable_relevance_sort()
        ctx.enable_similarity_filter()
        ctx.enable_estimation(random_seed=random_seed)
        ctx.enable_cache(cache_id)
        return ctx

    def register_model_config(self, model_config: ModelConfig) -> None:
        self._session_state.register_model_config(model_config)

    def enable_batching(self, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self._session_state.enable_batching(batch_size)

    def enable_early_stop(self) -> None:
        self._session_state.enable_early_stop()

    def enable_fusion(self) -> None:
        self._session_state.enable_fusion()

    def enable_minimal_provenance(self) -> None:
        self._session_state.enable_minimal_provenance()

    def enable_relevance_sort(self) -> None:
        self._session_state.enable_relevance_sort()

    def enable_similarity_filter(
        self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> None:
        self._session_state.enable_similarity_filter(similarity_threshold)

    def enable_estimation(
        self,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        relative_error: float = DEFAULT_RELATIVE_ERROR,
        random_seed: int | None = None,
    ) -> None:
        self._session_state.enable_estimation(
            confidence_level, relative_error, random_seed
        )

    def enable_cache(self, cache_id: str | None = None) -> None:
        """Enable caching (e.g., for language model responses).
        This must be called before any language model is registered.
        """
        self._session_state.enable_cache(cache_id)

    def read_rows(self, rows: list[Row], schema: Schema) -> DataFrame:
        table_provider = InMemoryTable(rows, schema)
        return self._create_data_frame(table_provider)

    def read_json(self, path: str, key: tuple[str, ...]) -> DataFrame:
        schema = self._infer_jsonl_schema(path, key)
        table_provider = JsonlTable(path, schema)
        return self._create_data_frame(table_provider)

    def read_pickle(self, path: str) -> DataFrame:
        with open(path, "rb") as f:
            rows, schema = pickle.load(f)
        return self.read_rows(rows, schema)

    def _create_data_frame(self, table_provider: TableProvider) -> DataFrame:
        plan = LogicalPlanBuilder.table_scan(table_provider).build()
        return DataFrame(plan, self._session_state)

    @staticmethod
    def _infer_jsonl_schema(path: str, key: tuple[str, ...]) -> Schema:
        with open(path, encoding="utf-8") as f:
            obj = json.loads(f.readline())

        fields = tuple(
            Field(name, cast(type[object], type(value))) for name, value in obj.items()
        )
        schema = Schema(fields, key)
        return schema
