from __future__ import annotations

from dataclasses import dataclass

from evergreen.catalog.schema import Field, Schema
from evergreen.common.constants import (
    EMBEDDING_FIELD_SUFFIX,
    SENTENCE_EMBEDDINGS_FIELD_SUFFIX,
)
from evergreen.data_frame import QueryResult
from evergreen.model.config import CortexModelConfig
from evergreen.planner.logical.plan import LogicalPlan
from evergreen.provenance import Prov
from evergreen.storage.row import AnnotatedValue, Row
from tests.conftest import (
    DEFAULT_CONNECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LANGUAGE_MODEL,
)


@dataclass(frozen=True)
class LogicalPlanPattern:
    plan_type: type[LogicalPlan]
    inputs: tuple[LogicalPlanPattern, ...] = ()

    def validate_match(self, plan: LogicalPlan) -> None:
        if not isinstance(plan, self.plan_type):
            raise AssertionError(
                f"Expected {self.plan_type.__name__}, "
                f"got {type(plan).__name__}:\n{plan.display()}"
            )

        actual_inputs = plan.inputs()
        if len(actual_inputs) != len(self.inputs):
            raise AssertionError(
                f"Expected {len(self.inputs)} input(s), "
                f"got {len(actual_inputs)}:\n{plan.display()}"
            )

        for pattern, actual in zip(self.inputs, actual_inputs, strict=True):
            pattern.validate_match(actual)


def P(plan_type: type[LogicalPlan], *inputs: LogicalPlanPattern) -> LogicalPlanPattern:
    """Shorthand for creating a `LogicalPlanPattern`."""
    return LogicalPlanPattern(plan_type, inputs)


def validate_check_result(
    result: QueryResult, is_valid: bool, prov: Prov, row_id: str = Row.AGG_ROW_ID
) -> None:
    rows = result.rows
    assert len(rows) == 1
    assert rows[0] == Row((row_id, is_valid))

    result_prov = rows[0].get_prov(1)

    assert isinstance(result_prov, Prov)
    assert result_prov == prov, f"{result_prov} != {prov}"


def add_embeddings(
    text_column_name: str, rows: list[Row], schema: Schema
) -> tuple[list[Row], Schema]:
    model_config = CortexModelConfig(
        [DEFAULT_LANGUAGE_MODEL], DEFAULT_EMBEDDING_MODEL, DEFAULT_CONNECTION_NAME
    )
    model = model_config.create_embedding_model(cache_dir=None)

    text_index = schema.index_of(text_column_name)
    embeddings = model.embed_with_sentences_batch(
        [str(row[text_index]) for row in rows]
    )

    new_rows = [
        row.with_values(
            (
                AnnotatedValue(doc_embedding, None),
                AnnotatedValue(sentence_embeddings, None),
            )
        )
        for row, (doc_embedding, sentence_embeddings) in zip(
            rows, embeddings, strict=True
        )
    ]

    new_schema = schema.with_fields(
        (
            Field(text_column_name + EMBEDDING_FIELD_SUFFIX, list[float]),
            Field(
                text_column_name + SENTENCE_EMBEDDINGS_FIELD_SUFFIX, list[list[float]]
            ),
        )
    )

    return new_rows, new_schema
