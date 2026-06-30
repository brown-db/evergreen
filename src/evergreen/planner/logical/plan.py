from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from evergreen.catalog.schema import Field, Schema
from evergreen.catalog.table import TableProvider
from evergreen.common.types import FusedPlanType
from evergreen.planner.logical.expr import Expr
from evergreen.planner.logical.types import EarlyStopComparison
from evergreen.storage.row import Row


class LogicalPlan(ABC):
    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def with_inputs(self, inputs: tuple[LogicalPlan, ...]) -> LogicalPlan:
        pass

    @abstractmethod
    def inputs(self) -> tuple[LogicalPlan, ...]:
        pass

    @abstractmethod
    def schema(self) -> Schema:
        pass

    def walk(self) -> Iterator[LogicalPlan]:
        """Pre-order traversal of this plan and all of its descendants."""
        yield self
        for input in self.inputs():
            yield from input.walk()

    def is_above(self, plan_type: type[LogicalPlan]) -> bool:
        return any(
            isinstance(input, plan_type) or input.is_above(plan_type)
            for input in self.inputs()
        )

    def is_or_above(self, plan_type: type[LogicalPlan]) -> bool:
        return isinstance(self, plan_type) or self.is_above(plan_type)

    def display(self, indent: int = 0) -> str:
        prefix = "  " * indent
        lines = [f"{prefix}{self!r}"]
        for input in self.inputs():
            lines.append(input.display(indent + 1))
        return "\n".join(lines)


@dataclass(frozen=True)
class TableScan(LogicalPlan):
    table_provider: TableProvider

    def __repr__(self) -> str:
        return "table_scan()"

    def with_inputs(self, inputs: tuple[LogicalPlan, ...]) -> LogicalPlan:
        return TableScan(self.table_provider)

    def inputs(self) -> tuple[LogicalPlan, ...]:
        return ()

    def schema(self) -> Schema:
        return self.table_provider.schema()


@dataclass(frozen=True)
class Filter(LogicalPlan):
    predicate: Expr
    input: LogicalPlan

    def __repr__(self) -> str:
        return f"filter({self.predicate!r})"

    def with_inputs(self, inputs: tuple[LogicalPlan, ...]) -> LogicalPlan:
        assert len(inputs) == 1
        return Filter(self.predicate, inputs[0])

    def inputs(self) -> tuple[LogicalPlan, ...]:
        return (self.input,)

    def schema(self) -> Schema:
        return self.input.schema()


@dataclass(frozen=True)
class Check(LogicalPlan):
    predicate: Expr
    input: LogicalPlan
    _schema: Schema = field(init=False)

    def __post_init__(self) -> None:
        input_schema = self.input.schema()
        object.__setattr__(
            self,
            "_schema",
            Schema(
                (*input_schema.key_fields(), Field(str(self.predicate), bool)),
                input_schema.key,
            ),
        )

    def __repr__(self) -> str:
        return f"check({self.predicate!r})"

    def with_inputs(self, inputs: tuple[LogicalPlan, ...]) -> LogicalPlan:
        assert len(inputs) == 1
        return Check(self.predicate, inputs[0])

    def inputs(self) -> tuple[LogicalPlan, ...]:
        return (self.input,)

    def schema(self) -> Schema:
        return self._schema


@dataclass(frozen=True)
class Projection(LogicalPlan):
    exprs: tuple[Expr, ...]
    input: LogicalPlan
    _schema: Schema = field(init=False)

    def __post_init__(self) -> None:
        input_schema = self.input.schema()
        fields = tuple(expr.to_field(input_schema) for expr in self.exprs)
        object.__setattr__(self, "_schema", Schema(fields, input_schema.key))

    def __repr__(self) -> str:
        return f"projection([{', '.join(repr(expr) for expr in self.exprs)}])"

    def with_inputs(self, inputs: tuple[LogicalPlan, ...]) -> LogicalPlan:
        assert len(inputs) == 1
        return Projection(self.exprs, inputs[0])

    def inputs(self) -> tuple[LogicalPlan, ...]:
        return (self.input,)

    def schema(self) -> Schema:
        return self._schema


@dataclass(frozen=True)
class FusedFilterProjection(LogicalPlan):
    exprs: tuple[Expr, ...]
    plan_types: tuple[FusedPlanType, ...]
    input: LogicalPlan
    _schema: Schema = field(init=False)

    def __post_init__(self) -> None:
        input_schema = self.input.schema()
        # Schema includes only PROJECTION expressions
        fields = tuple(
            expr.to_field(input_schema)
            for expr, plan_type in zip(self.exprs, self.plan_types, strict=True)
            if plan_type == FusedPlanType.PROJECTION
        )
        object.__setattr__(self, "_schema", Schema(fields, input_schema.key))

    def __repr__(self) -> str:
        return (
            "fused_filter_projection("
            "exprs="
            f"[{', '.join(repr(expr) for expr in self.exprs)}], "
            "plan_types="
            f"[{', '.join(plan_type.name.lower() for plan_type in self.plan_types)}]"
            ")"
        )

    def with_inputs(self, inputs: tuple[LogicalPlan, ...]) -> LogicalPlan:
        assert len(inputs) == 1
        return FusedFilterProjection(self.exprs, self.plan_types, inputs[0])

    def inputs(self) -> tuple[LogicalPlan, ...]:
        return (self.input,)

    def schema(self) -> Schema:
        return self._schema


@dataclass(frozen=True)
class Aggregate(LogicalPlan):
    agg_exprs: tuple[Expr, ...]
    group_exprs: tuple[Expr, ...]
    early_stop_comparisons: tuple[EarlyStopComparison | None, ...]
    input: LogicalPlan
    _schema: Schema = field(init=False)

    def __post_init__(self) -> None:
        input_schema = self.input.schema()

        if self.group_exprs:
            fields = [expr.to_field(input_schema) for expr in self.group_exprs]
        else:
            fields = [Field(Row.INTERNAL_ID_FIELD_NAME, str)]

        key = tuple(field.name for field in fields)
        fields.extend(expr.to_field(input_schema) for expr in self.agg_exprs)
        object.__setattr__(self, "_schema", Schema(tuple(fields), key))

    def __repr__(self) -> str:
        return (
            f"aggregate([{', '.join(repr(expr) for expr in self.agg_exprs)}], "
            f"group_by=[{', '.join(repr(expr) for expr in self.group_exprs)}])"
        )

    def with_inputs(self, inputs: tuple[LogicalPlan, ...]) -> LogicalPlan:
        assert len(inputs) == 1
        return Aggregate(
            self.agg_exprs, self.group_exprs, self.early_stop_comparisons, inputs[0]
        )

    def inputs(self) -> tuple[LogicalPlan, ...]:
        return (self.input,)

    def schema(self) -> Schema:
        return self._schema


@dataclass(frozen=True)
class Sort(LogicalPlan):
    exprs: tuple[Expr, ...]
    descending: bool
    input: LogicalPlan

    def __repr__(self) -> str:
        return (
            f"sort([{', '.join(repr(expr) for expr in self.exprs)}], "
            f"descending={self.descending})"
        )

    def with_inputs(self, inputs: tuple[LogicalPlan, ...]) -> LogicalPlan:
        assert len(inputs) == 1
        return Sort(self.exprs, self.descending, inputs[0])

    def inputs(self) -> tuple[LogicalPlan, ...]:
        return (self.input,)

    def schema(self) -> Schema:
        return self.input.schema()


@dataclass(frozen=True)
class RelevanceSort(LogicalPlan):
    text_column_name: str
    query_text: str
    query_embedding: list[float]
    inclusion_keywords: list[str]
    exclusion_keywords: list[str]
    input: LogicalPlan

    def __repr__(self) -> str:
        return (
            "relevance_sort("
            f"text_column_name={self.text_column_name!r}, "
            f"query_text={self.query_text!r}, "
            f"inclusion_keywords={self.inclusion_keywords}, "
            f"exclusion_keywords={self.exclusion_keywords})"
        )

    def with_inputs(self, inputs: tuple[LogicalPlan, ...]) -> LogicalPlan:
        assert len(inputs) == 1
        return RelevanceSort(
            self.text_column_name,
            self.query_text,
            self.query_embedding,
            self.inclusion_keywords,
            self.exclusion_keywords,
            inputs[0],
        )

    def inputs(self) -> tuple[LogicalPlan, ...]:
        return (self.input,)

    def schema(self) -> Schema:
        return self.input.schema()


@dataclass(frozen=True)
class WithRank(LogicalPlan):
    expr: Expr
    descending: bool
    input: LogicalPlan
    _schema: Schema = field(init=False)

    def __post_init__(self) -> None:
        input_schema = self.input.schema()
        object.__setattr__(
            self, "_schema", input_schema.with_fields((Field("rank", int),))
        )

    def __repr__(self) -> str:
        return f"with_rank({self.expr!r}, descending={self.descending})"

    def with_inputs(self, inputs: tuple[LogicalPlan, ...]) -> LogicalPlan:
        assert len(inputs) == 1
        return WithRank(self.expr, self.descending, inputs[0])

    def inputs(self) -> tuple[LogicalPlan, ...]:
        return (self.input,)

    def schema(self) -> Schema:
        return self._schema


@dataclass(frozen=True)
class CountScan(LogicalPlan):
    input: LogicalPlan

    def __repr__(self) -> str:
        return "count_scan()"

    def with_inputs(self, inputs: tuple[LogicalPlan, ...]) -> LogicalPlan:
        assert len(inputs) == 1
        return CountScan(inputs[0])

    def inputs(self) -> tuple[LogicalPlan, ...]:
        return (self.input,)

    def schema(self) -> Schema:
        return self.input.schema()


@dataclass(frozen=True)
class Shuffle(LogicalPlan):
    exprs: tuple[Expr, ...]
    shuffle_rows: bool  # whether to shuffle the base rows
    seed: int | None
    input: LogicalPlan

    def __repr__(self) -> str:
        return (
            f"shuffle([{', '.join(repr(expr) for expr in self.exprs)}], "
            f"shuffle_rows={self.shuffle_rows}, seed={self.seed})"
        )

    def with_inputs(self, inputs: tuple[LogicalPlan, ...]) -> LogicalPlan:
        assert len(inputs) == 1
        return Shuffle(self.exprs, self.shuffle_rows, self.seed, inputs[0])

    def inputs(self) -> tuple[LogicalPlan, ...]:
        return (self.input,)

    def schema(self) -> Schema:
        return self.input.schema()


@dataclass(frozen=True)
class Log(LogicalPlan):
    """Log operator that writes intermediate results to a pickle file.

    This is used to checkpoint intermediate results after semantic operations,
    which are used to evaluate the quality of the semantic operations.

    TODO: Update the optimizer to correctly rewrite plans when the log operator
    is present. Currently, the optimizer may not rewrite the plan correctly when
    the log operator is placed in arbitrary points in the plan. However, this is
    not an issue when the log operator is placed after the semantic operations
    but before the aggregation.
    """

    path: Path
    input: LogicalPlan

    def __repr__(self) -> str:
        return f"log(path={str(self.path)!r})"

    def with_inputs(self, inputs: tuple[LogicalPlan, ...]) -> LogicalPlan:
        assert len(inputs) == 1
        return Log(self.path, inputs[0])

    def inputs(self) -> tuple[LogicalPlan, ...]:
        return (self.input,)

    def schema(self) -> Schema:
        return self.input.schema()
