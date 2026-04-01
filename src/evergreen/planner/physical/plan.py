from __future__ import annotations

import logging
import pickle
import random
from abc import ABC, abstractmethod
from collections import Counter, deque
from itertools import groupby
from typing import Protocol, cast

import numpy as np

from evergreen.catalog.schema import Schema
from evergreen.common.constants import EMBEDDING_FIELD_SUFFIX
from evergreen.common.types import FusedPlanType
from evergreen.data_source import DataSource
from evergreen.planner.logical.types import EarlyStopComparison, Operator
from evergreen.planner.physical.accumulator import Accumulator
from evergreen.planner.physical.expr import (
    AggregateFunctionExpr,
    BinaryExpr,
    FusedPrompt,
    PhysicalExpr,
    Prompt,
)
from evergreen.provenance import Prov
from evergreen.retrieval import rank_by_rrf
from evergreen.storage.row import AnnotatedValue, Row, RowId

logger = logging.getLogger(__name__)


class SkipPredicate(Protocol):
    def __call__(self, row: Row, schema: Schema) -> bool: ...


class CountProvider(Protocol):
    """Provides row and group counts for early stopping in aggregates.

    Implemented by Sort (for grouped aggregates) and CountScan (for ungrouped
    aggregates) to provide precomputed totals for early stopping.
    """

    def row_count(self, group_prefix: tuple[object, ...]) -> int:
        """Total rows matching the group prefix.

        Args:
            group_prefix: Filter to rows in groups starting with this group prefix.
                Empty tuple means all rows.

        Example with Sort(restaurant, dish):
            row_count(("McD", "BigMac"))  -> 5  (reviews for BigMac at McD)
            row_count(("McD",))           -> 10 (all McD reviews)
            row_count(())                 -> 30 (all reviews)
        """
        ...

    def group_count(
        self, group_prefix: tuple[object, ...], group_key_length: int
    ) -> int:
        """Count unique groups of the specified group key length matching the
        group prefix.

        This is used when an aggregate is above another grouped aggregate.
        For example, if the outer aggregate has no GROUP BY but the inner
        aggregate has GROUP BY (e.g., business_id), the outer needs to know how
        many unique business_ids exist.

        Args:
            group_prefix: Filter to groups starting with this group prefix.
            group_key_length: Number of columns that define a group (i.e., the number
                of GROUP BY columns in the input aggregate).

        Example with Sort(restaurant, dish):
            group_count((), 1)        -> 3  (total restaurants)
            group_count((), 2)        -> 6  (total dishes across all restaurants)
            group_count(("McD",), 2)  -> 2  (total dishes at McDonald's)
        """
        ...


class PhysicalPlan(ABC):
    def __init__(self) -> None:
        self._skip_predicates: list[SkipPredicate] = []

    @abstractmethod
    def inputs(self) -> tuple[PhysicalPlan, ...]:
        pass

    @abstractmethod
    def schema(self) -> Schema:
        pass

    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def next(self) -> Row | None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def add_skip_predicate(self, skip_predicate: SkipPredicate) -> None:
        self._skip_predicates.append(skip_predicate)

    def should_skip(self, row: Row, schema: Schema) -> bool:
        return any(
            skip_predicate(row, schema) for skip_predicate in self._skip_predicates
        )

    def is_above(self, plan_type: type[PhysicalPlan]) -> bool:
        return any(
            isinstance(input, plan_type) or input.is_above(plan_type)
            for input in self.inputs()
        )

    @staticmethod
    def _get_id(
        row: Row, schema: Schema, exprs: tuple[PhysicalExpr, ...]
    ) -> tuple[object, ...]:
        row_id = row.id(schema)
        return tuple(expr.evaluate(row, row_id).value for expr in exprs)


class BatchedPhysicalPlan(PhysicalPlan, ABC):
    def __init__(self, input: PhysicalPlan, batch_size: int) -> None:
        super().__init__()
        self._input = input
        self._batch_size = batch_size
        self._buffer: deque[Row] = deque()
        self._input_exhausted: bool = False

    def inputs(self) -> tuple[PhysicalPlan, ...]:
        return (self._input,)

    def open(self) -> None:
        self._input.open()

    def next(self) -> Row | None:
        while not self._buffer:
            if self._input_exhausted:
                return None
            self._buffer = deque(self._process_batch())
        return self._buffer.popleft()

    def close(self) -> None:
        self._input.close()

    @abstractmethod
    def _process_batch(self) -> list[Row]:
        pass

    def _pull_batch(self) -> list[tuple[Row, RowId]]:
        rows: list[tuple[Row, RowId]] = []
        input_schema = self._input.schema()
        while len(rows) < self._batch_size:
            row = self._input.next()
            if row is None:
                self._input_exhausted = True
                break
            if not self.should_skip(row, input_schema):
                rows.append((row, row.id(input_schema)))
        return rows


class TableScan(PhysicalPlan):
    def __init__(self, data_source: DataSource, schema: Schema) -> None:
        super().__init__()
        self._data_source = data_source
        self._schema = schema

    def inputs(self) -> tuple[PhysicalPlan, ...]:
        return ()

    def schema(self) -> Schema:
        return self._schema

    def open(self) -> None:
        self._data_source.open()

    def next(self) -> Row | None:
        while (row := self._data_source.next()) is not None:
            if self.should_skip(row, self._schema):
                continue
            return row
        return None

    def close(self) -> None:
        self._data_source.close()


class Filter(BatchedPhysicalPlan):
    def __init__(
        self,
        predicate: PhysicalExpr,
        input: PhysicalPlan,
        schema: Schema,
        batch_size: int,
    ) -> None:
        super().__init__(input, batch_size)
        self._predicate = predicate
        self._schema = schema

    def schema(self) -> Schema:
        return self._schema

    def _process_batch(self) -> list[Row]:
        batch = self._pull_batch()

        if not batch:
            return []

        batch_values = self._predicate.evaluate_batch(batch)
        selected_rows: list[Row] = []
        for (row, row_id), annotated_value in zip(batch, batch_values, strict=True):
            if annotated_value.value:
                selected_rows.append(row)
            else:
                logger.debug(
                    "Row %s filtered out by predicate %s", row_id, self._predicate
                )

        return selected_rows


class Check(PhysicalPlan):
    def __init__(
        self, predicate: PhysicalExpr, input: PhysicalPlan, schema: Schema
    ) -> None:
        super().__init__()
        self._predicate = predicate
        self._input = input
        self._schema = schema

    def inputs(self) -> tuple[PhysicalPlan, ...]:
        return (self._input,)

    def schema(self) -> Schema:
        return self._schema

    def open(self) -> None:
        self._input.open()

    def next(self) -> Row | None:
        while (row := self._input.next()) is not None:
            if self.should_skip(row, self._input.schema()):
                continue

            row_id = row.id(self._input.schema())
            annotated_value = self._predicate.evaluate(row, row_id)
            return Row(
                (
                    *tuple(AnnotatedValue(value, None) for value in row_id),
                    annotated_value,
                )
            )

        return None

    def close(self) -> None:
        self._input.close()


class Projection(BatchedPhysicalPlan):
    def __init__(
        self,
        exprs: tuple[PhysicalExpr, ...],
        input: PhysicalPlan,
        schema: Schema,
        batch_size: int,
    ) -> None:
        super().__init__(input, batch_size)
        self._exprs = exprs
        self._schema = schema

    def schema(self) -> Schema:
        return self._schema

    def _process_batch(self) -> list[Row]:
        batch = self._pull_batch()

        if not batch:
            return []

        batch_values = [expr.evaluate_batch(batch) for expr in self._exprs]

        return [Row(row_values) for row_values in zip(*batch_values, strict=True)]


class FusedFilterProjection(BatchedPhysicalPlan):
    def __init__(
        self,
        exprs: tuple[PhysicalExpr, ...],
        plan_types: tuple[FusedPlanType, ...],
        input: PhysicalPlan,
        schema: Schema,
        batch_size: int,
    ) -> None:
        super().__init__(input, batch_size)
        self._exprs = exprs
        self._plan_types = plan_types
        self._schema = schema

        # Separate prompts vs non-prompts
        self._prompt_indices: list[int] = []
        self._non_prompt_indices: list[int] = []
        prompts: list[Prompt] = []

        for i, expr in enumerate(self._exprs):
            if isinstance(expr, Prompt):
                self._prompt_indices.append(i)
                prompts.append(expr)
            else:
                self._non_prompt_indices.append(i)

        # Created FusedPrompt if worth batching
        self._fused_prompt: FusedPrompt | None = None
        if len(prompts) >= 2:
            self._fused_prompt = FusedPrompt(
                prompt_strs=tuple(prompt.prompt_str() for prompt in prompts),
                return_types=tuple(prompt.return_type() for prompt in prompts),
                field_indices=self._merge_field_indices(prompts),
                model=prompts[0].model(),
            )

    @staticmethod
    def _merge_field_indices(prompts: list[Prompt]) -> tuple[tuple[str, int], ...]:
        field_indices: list[tuple[str, int]] = []
        seen: set[str] = set()
        for prompt in prompts:
            for field_name, index in prompt.field_indices():
                if field_name not in seen:
                    field_indices.append((field_name, index))
                    seen.add(field_name)
        return tuple(field_indices)

    def schema(self) -> Schema:
        return self._schema

    def _process_batch(self) -> list[Row]:
        batch = self._pull_batch()

        if not batch:
            return []

        # batch_values[expr_index][row_index]
        batch_values: list[list[AnnotatedValue]] = [[] for _ in self._exprs]

        if self._fused_prompt is not None:
            fused_results = self._fused_prompt.evaluate_batch(batch)
            for i, index in enumerate(self._prompt_indices):
                batch_values[index] = [fr[i] for fr in fused_results]
        else:
            for index in self._prompt_indices:
                batch_values[index] = self._exprs[index].evaluate_batch(batch)

        for index in self._non_prompt_indices:
            batch_values[index] = self._exprs[index].evaluate_batch(batch)

        output: list[Row] = []
        for r, (_, row_id) in enumerate(batch):
            if not all(
                batch_values[i][r].value
                for i, plan_type in enumerate(self._plan_types)
                if plan_type == FusedPlanType.FILTER
            ):
                logger.debug(
                    "Row %s filtered out by fused prompt %s",
                    row_id,
                    self._fused_prompt,
                )
                continue
            output.append(
                Row(
                    tuple(
                        batch_values[i][r]
                        for i, plan_type in enumerate(self._plan_types)
                        if plan_type == FusedPlanType.PROJECTION
                    )
                )
            )

        return output


class StreamAggregate(PhysicalPlan):
    __match_args__ = (
        "_agg_exprs",
        "_group_exprs",
        "_early_stop_comparisons",
        "_early_stop_enabled",
        "_input",
        "_schema",
    )

    def __init__(
        self,
        agg_exprs: tuple[AggregateFunctionExpr, ...],
        group_exprs: tuple[PhysicalExpr, ...],
        early_stop_comparisons: tuple[EarlyStopComparison | None, ...],
        early_stop_enabled: bool,
        input: PhysicalPlan,
        schema: Schema,
    ) -> None:
        super().__init__()
        self._agg_exprs = agg_exprs
        self._group_exprs = group_exprs
        self._early_stop_comparisons = early_stop_comparisons
        self._early_stop_enabled = early_stop_enabled
        self._input = input
        self._schema = schema

        self._accumulators: tuple[Accumulator, ...] = ()
        self._current_group_id: tuple[object, ...] = ()
        self._current_group_count: int = 0
        self._pending_row: Row | None = None
        self._done = False

        self._skip_group_id: tuple[object, ...] | None = None
        self._group_field_names = tuple(str(expr) for expr in self._group_exprs)
        self._count_provider: CountProvider | None = None
        self._input_group_key_length: int = 0
        self._confidence_level: float | None = None

    def inputs(self) -> tuple[PhysicalPlan, ...]:
        return (self._input,)

    def schema(self) -> Schema:
        return self._schema

    def in_skip_group(self, row: Row, schema: Schema) -> bool:
        if self._skip_group_id is None:
            return False

        schema_field_names = set(schema.field_names())
        return all(
            name in schema_field_names and value == row[schema.index_of(name)]
            for name, value in zip(
                self._group_field_names, self._skip_group_id, strict=True
            )
        )

    def set_count_provider(
        self, count_provider: CountProvider, input_group_key_length: int
    ) -> None:
        """Sets the count provider for the aggregate operator.

        This should be called during the planning phase to enable precomputed
        totals for early stopping.

        Args:
            count_provider: The Sort or CountScan that provides counts.
            input_group_key_length: Number of GROUP BY columns in the input aggregate.
                This is 0 if the input is not a grouped aggregate
                (e.g., directly from Sort/TableScan).

                For nested aggregates like:
                    Outer (no GROUP BY)
                        |
                    Inner (GROUP BY business_id)
                        |
                    Sort (business_id)

                The Outer aggregate would have input_group_key_length=1
                because Inner has 1 GROUP BY column. This tells
                Outer to count groups (businesses).
        """
        self._count_provider = count_provider
        self._input_group_key_length = input_group_key_length

    def set_confidence_level(self, confidence_level: float) -> None:
        """Sets the confidence level for the aggregate operator.
        This should be called during the planning phase.
        """
        self._confidence_level = confidence_level

    def open(self) -> None:
        self._input.open()

    def next(self) -> Row | None:
        if self._done:
            return None

        if self._group_exprs:
            return self._next_with_group_by()

        return self._next_without_group_by()

    def _next_without_group_by(self) -> Row | None:
        # Call next() before creating accumulators to precompute the total in CountScan
        row = self._input.next()

        if not self._accumulators:
            self._accumulators = self._create_accumulators()

        while row is not None:
            if self.should_skip(row, self._input.schema()):
                row = self._input.next()
                continue
            self._update_accumulators(row)
            if self._can_stop():
                break
            row = self._input.next()

        self._done = True
        return self._emit_result(group_id=())

    def _next_with_group_by(self) -> Row | None:
        # We assume that the input is sorted by the group ID
        # for the following logic to work

        # Process pending row from previous call,
        # which started a new group
        if self._pending_row:
            if self.should_skip(self._pending_row, self._input.schema()):
                self._current_group_id = ()
                self._accumulators = ()
            else:
                self._update_accumulators(self._pending_row)
            self._pending_row = None

        while (row := self._input.next()) is not None:
            if self.should_skip(row, self._input.schema()):
                continue

            group_id = self._get_id(row, self._input.schema(), self._group_exprs)

            # New group: emit accumulated result and start fresh
            if self._current_group_id and self._current_group_id != group_id:
                result = self._emit_result(self._current_group_id)
                self._current_group_id = group_id
                self._current_group_count += 1
                self._pending_row = row
                self._accumulators = self._create_accumulators()
                return result

            # First row or same group: accumulate
            if not self._current_group_id:
                self._current_group_id = group_id
                self._current_group_count += 1
                if not self._accumulators:
                    self._accumulators = self._create_accumulators()
            self._update_accumulators(row)

            # Can stop early: emit the accumulated result and start fresh
            # Group ID and accumulators will be created in the next call
            # via the if statement above.
            if self._can_stop():
                self._skip_group_id = self._current_group_id
                result = self._emit_result(self._current_group_id)
                self._current_group_id = ()
                self._accumulators = ()
                return result

        self._done = True

        # Emit final group if any
        if self._current_group_id:
            return self._emit_result(self._current_group_id)

        # No input rows
        return None

    def _can_stop(self) -> bool:
        if not self._early_stop_enabled:
            return False

        # TODO: Currently, we only support AND semantics for early stopping.
        # Ideally, we should also support OR semantics, where we stop when any
        # accumulator can stop (assuming OR is used in a downstream predicate).
        for comparison, accumulator in zip(
            self._early_stop_comparisons, self._accumulators, strict=True
        ):
            if not accumulator.can_stop(comparison):
                return False
        return True

    def _update_accumulators(self, row: Row) -> None:
        row_id = row.id(self._input.schema())
        for expr, accumulator in zip(self._agg_exprs, self._accumulators, strict=True):
            annotated_values = expr.evaluate_args(row, row_id)
            accumulator.update(annotated_values)

    def _create_accumulators(self) -> tuple[Accumulator, ...]:
        # Set accumulator confidence levels
        if self._confidence_level is None:
            accumulator_confidence_level = None
        else:
            alpha = 1 - self._confidence_level

            # Count the number of aggregate functions that support estimation
            num_agg_exprs_supporting_estimation = sum(
                1 for expr in self._agg_exprs if expr.supports_estimation()
            )

            # Planner only sets confidence level if at least one aggregation
            # function supports estimation
            assert num_agg_exprs_supporting_estimation > 0

            if self._count_provider is not None:
                # Control for FWER with Bonferroni correction when group counts
                # are known. We count total groups at this aggregate's level.
                group_count = self._count_provider.group_count(
                    group_prefix=(), group_key_length=len(self._current_group_id)
                )
                adjusted_alpha = alpha / (
                    num_agg_exprs_supporting_estimation * group_count
                )
            elif self._group_exprs:
                # Grouped aggregate without `TotalCountProvider`.
                # This can happen when operator fusion is enabled and none of the
                # fused operators depend on the group keys. This means that we
                # push Sort (i.e., the count provider) below the fused operators.
                # Since Sort is below FusedFilterProjection, it is below a
                # filter, and thus does not provide a valid count.
                # In this case, when group counts are not known, we
                # control FWER with alpha spending (geometric decay).
                group_count = self._current_group_count
                adjusted_alpha = alpha / (
                    num_agg_exprs_supporting_estimation * 2**self._current_group_count
                )
            else:
                # Ungrouped aggregate without `CountScan` count provider.
                # This occurs when operator fusion is enabled.
                group_count = 1
                adjusted_alpha = alpha / num_agg_exprs_supporting_estimation

            accumulator_confidence_level = 1 - adjusted_alpha

            logger.debug(
                "aggregate confidence level allocation: "
                "agg_exprs=[%s], "
                "group_exprs=[%s], "
                "confidence_level=%f, "
                "num_agg_exprs_supporting_estimation=%d, "
                "group_count=%d, "
                "accumulator_confidence_level=%f",
                ", ".join(str(expr) for expr in self._agg_exprs),
                ", ".join(str(expr) for expr in self._group_exprs),
                self._confidence_level,
                num_agg_exprs_supporting_estimation,
                group_count,
                accumulator_confidence_level,
            )

        # Set precomputed totals
        if self._count_provider is None:
            precomputed_total = None
        elif self._input_group_key_length > 0:
            # Input is a grouped aggregate - count groups at input's grouping level
            # that match our current group prefix
            precomputed_total = self._count_provider.group_count(
                group_prefix=self._current_group_id,
                group_key_length=self._input_group_key_length,
            )
        else:
            # Input is raw rows (from Sort/TableScan) - count rows matching our group
            precomputed_total = self._count_provider.row_count(
                group_prefix=self._current_group_id
            )

        return tuple(
            expr.create_accumulator(
                precomputed_total,
                accumulator_confidence_level if expr.supports_estimation() else None,
            )
            for expr in self._agg_exprs
        )

    def _emit_result(self, group_id: tuple[object, ...]) -> Row:
        assert all(not isinstance(value, AnnotatedValue) for value in group_id)

        if group_id:
            result = [AnnotatedValue(value, None) for value in group_id]
        else:
            result = [AnnotatedValue(Row.AGG_ROW_ID, None)]

        result.extend(accumulator.evaluate() for accumulator in self._accumulators)

        return Row(tuple(result))

    def close(self) -> None:
        self._input.close()


class Sort(PhysicalPlan):
    def __init__(
        self,
        exprs: tuple[PhysicalExpr, ...],
        descending: bool,
        input: PhysicalPlan,
        schema: Schema,
    ) -> None:
        super().__init__()
        self._exprs = exprs
        self._descending = descending
        self._input = input
        self._schema = schema
        self._sorted_rows: list[Row] | None = None
        self._index: int = 0
        # Counts of rows for each group ID defined by the exprs
        self._group_counts: Counter[tuple[object, ...]] = Counter()

    def inputs(self) -> tuple[PhysicalPlan, ...]:
        return (self._input,)

    def schema(self) -> Schema:
        return self._schema

    def open(self) -> None:
        self._input.open()

    def next(self) -> Row | None:
        if self._sorted_rows is None:
            self._sorted_rows = []
            input_schema = self._input.schema()
            while (row := self._input.next()) is not None:
                if self.should_skip(row, input_schema):
                    continue
                self._sorted_rows.append(row)
                group_id = self._get_id(row, input_schema, self._exprs)
                self._group_counts[group_id] += 1

            self._sorted_rows.sort(
                key=lambda row: self._get_id(row, input_schema, self._exprs),
                reverse=self._descending,
            )

        while self._index < len(self._sorted_rows):
            row = self._sorted_rows[self._index]
            self._index += 1
            if not self.should_skip(row, self._input.schema()):
                return row

        return None

    def close(self) -> None:
        self._input.close()

    def row_count(self, group_prefix: tuple[object, ...]) -> int:
        """Total rows matching the group prefix.

        If group prefix matches a key exactly, returns that key's count.
        Otherwise, sums counts for all keys that start with the group prefix.
        """
        if group_prefix in self._group_counts:
            return self._group_counts[group_prefix]

        group_prefix_len = len(group_prefix)
        return sum(
            self._group_counts[key]
            for key in self._group_counts
            if key[:group_prefix_len] == group_prefix
        )

    def group_count(
        self, group_prefix: tuple[object, ...], group_key_length: int
    ) -> int:
        """Count unique groups of the specified group key length matching the
        group prefix.

        This counts how many distinct group keys of length `group_key_length` exist
        among all keys that start with `group_prefix`.

        Args:
            group_prefix: Filter to groups starting with this group prefix.
            group_key_length: Number of columns that define a group.

        Returns:
            Number of unique groups at the specified level within the prefix.
        """
        group_prefix_len = len(group_prefix)
        return len(
            set(
                key[:group_key_length]
                for key in self._group_counts
                if key[:group_prefix_len] == group_prefix
            )
        )


class RelevanceSort(PhysicalPlan):
    def __init__(
        self,
        text_column_name: str,
        query_embedding: list[float],
        inclusion_keywords: list[str],
        exclusion_keywords: list[str],
        input: PhysicalPlan,
        schema: Schema,
    ) -> None:
        super().__init__()
        self._text_column_name = text_column_name
        self._query_embedding = query_embedding
        self._inclusion_keywords = inclusion_keywords
        self._exclusion_keywords = exclusion_keywords
        self._input = input
        self._schema = schema
        self._sorted_rows: list[Row] | None = None
        self._index: int = 0

    def inputs(self) -> tuple[PhysicalPlan, ...]:
        return (self._input,)

    def schema(self) -> Schema:
        return self._schema

    def open(self) -> None:
        self._input.open()

    def next(self) -> Row | None:
        if self._sorted_rows is None:
            self._sorted_rows = self._buffer_and_sort()

        while self._index < len(self._sorted_rows):
            row = self._sorted_rows[self._index]
            self._index += 1
            if not self.should_skip(row, self._input.schema()):
                return row

        return None

    def _buffer_and_sort(self) -> list[Row]:
        """Buffer all input rows and sort by Reciprocal Rank Fusion (RRF)
        relevance score.
        """

        input_schema = self._input.schema()
        column_index = input_schema.index_of(self._text_column_name)
        embedding_index = input_schema.index_of(
            self._text_column_name + EMBEDDING_FIELD_SUFFIX
        )

        rows: list[Row] = []
        docs: list[str] = []
        doc_embeddings: list[list[float]] = []

        # Buffer rows and extract embeddings and texts
        while (row := self._input.next()) is not None:
            if self.should_skip(row, input_schema):
                continue
            rows.append(row)
            docs.append(str(row[column_index]))
            doc_embeddings.append(cast(list[float], row[embedding_index]))

        if not rows:
            return []

        sorted_indices = rank_by_rrf(
            docs,
            np.array(doc_embeddings),
            np.array(self._query_embedding),
            self._inclusion_keywords,
            self._exclusion_keywords,
        )

        return [rows[i] for i in sorted_indices]

    def close(self) -> None:
        self._input.close()


class WithRank(PhysicalPlan):
    def __init__(
        self,
        expr: PhysicalExpr,
        descending: bool,
        input: PhysicalPlan,
        schema: Schema,
        minimal_provenance: bool,
    ) -> None:
        super().__init__()
        self._expr = expr
        self._descending = descending
        self._input = input
        self._schema = schema
        self._minimal_provenance = minimal_provenance
        self._ranked_rows: list[Row] | None = None
        self._index: int = 0

    def inputs(self) -> tuple[PhysicalPlan, ...]:
        return (self._input,)

    def schema(self) -> Schema:
        return self._schema

    def open(self) -> None:
        self._input.open()

    def next(self) -> Row | None:
        if self._ranked_rows is None:
            rows: list[Row] = []
            while (row := self._input.next()) is not None:
                if self.should_skip(row, self._input.schema()):
                    continue
                rows.append(row)

            input_schema = self._input.schema()

            # Pre-compute scores as `AnnotatedValues` to preserve `ProvCollection`
            scored_rows: list[tuple[Row, AnnotatedValue]] = []
            for row in rows:
                score_av = self._expr.evaluate(row, row.id(input_schema))
                scored_rows.append((row, score_av))

            # Sort by score to compute ranks
            scored_rows.sort(key=lambda r: (r[1].value,), reverse=self._descending)

            # Assign dense ranks
            ranks: list[int] = []
            prev_score = None
            rank = 0
            for _, score_av in scored_rows:
                if score_av.value != prev_score:
                    rank += 1
                    prev_score = score_av.value
                ranks.append(rank)

            # Compute provenance for each row
            self._ranked_rows = []
            for i, (row, score_av) in enumerate(scored_rows):
                rank = ranks[i]

                # Provenance is the product of all pairwise comparisons
                prov = Prov.one()

                for j, (_, other_score_av) in enumerate(scored_rows):
                    if i == j:
                        continue

                    other_rank = ranks[j]

                    # Determine relationship based on ranks
                    if rank < other_rank:
                        op = Operator.GT if self._descending else Operator.LT
                    elif rank == other_rank:
                        op = Operator.EQ
                    else:
                        op = Operator.LT if self._descending else Operator.GT

                    # Compute provenance for pairwise comparison
                    result = BinaryExpr.evaluate_comparison(
                        score_av,
                        op,
                        other_score_av,
                        row_id=None,
                        predicate=None,
                        minimal_provenance=self._minimal_provenance,
                    )

                    # Multiply provenance (AND in semiring)
                    assert isinstance(result.prov, Prov)
                    prov *= result.prov

                ranked_row = row.with_values((AnnotatedValue(rank, prov),))
                self._ranked_rows.append(ranked_row)

        while self._index < len(self._ranked_rows):
            row = self._ranked_rows[self._index]
            self._index += 1
            if not self.should_skip(row, self._schema):
                return row

        return None

    def close(self) -> None:
        self._input.close()


class CountScan(PhysicalPlan):
    def __init__(self, input: PhysicalPlan, schema: Schema) -> None:
        super().__init__()
        self._input = input
        self._schema = schema
        self._rows: list[Row] | None = None
        self._index: int = 0

    def inputs(self) -> tuple[PhysicalPlan, ...]:
        return (self._input,)

    def schema(self) -> Schema:
        return self._schema

    def open(self) -> None:
        self._input.open()

    def next(self) -> Row | None:
        if self._rows is None:
            self._rows = []
            while (row := self._input.next()) is not None:
                if self.should_skip(row, self._input.schema()):
                    continue
                self._rows.append(row)

        while self._index < len(self._rows):
            row = self._rows[self._index]
            self._index += 1
            if not self.should_skip(row, self._input.schema()):
                return row

        return None

    def close(self) -> None:
        self._input.close()

    def row_count(self, group_prefix: tuple[object, ...]) -> int:
        """Total rows buffered by CountScan.

        CountScan has no grouping structure, so it always returns the total
        count regardless of group prefix.
        """
        assert self._rows is not None
        return len(self._rows)

    def group_count(
        self, group_prefix: tuple[object, ...], group_key_length: int
    ) -> int:
        """CountScan has no grouping structure.

        This should only be called for ungrouped aggregates where there's
        no nested grouped aggregate below. Returns 1 as there's conceptually
        one "group" (all rows).
        """
        return 1


class Shuffle(PhysicalPlan):
    """Hierarchically shuffle rows at all group levels.

    Ensures exchangeable data order for confidence sequence validity.
    Groups remain contiguous after shuffling for StreamAggregate compatibility.

    When shuffle_rows=False, row order within the finest groups is preserved.
    This is used when RelevanceSort is present to maintain relevance ordering
    for deterministic early stopping at the innermost aggregate level.
    """

    def __init__(
        self,
        exprs: tuple[PhysicalExpr, ...],
        shuffle_rows: bool,
        seed: int | None,
        input: PhysicalPlan,
        schema: Schema,
    ) -> None:
        super().__init__()
        self._exprs = exprs
        self._shuffle_rows = shuffle_rows
        self._seed = seed
        self._input = input
        self._schema = schema
        self._shuffled_rows: list[Row] | None = None
        self._index: int = 0
        self._rng = random.Random(seed)

    def inputs(self) -> tuple[PhysicalPlan, ...]:
        return (self._input,)

    def schema(self) -> Schema:
        return self._schema

    def open(self) -> None:
        self._input.open()

    def next(self) -> Row | None:
        if self._shuffled_rows is None:
            rows: list[Row] = []
            while (row := self._input.next()) is not None:
                if self.should_skip(row, self._input.schema()):
                    continue
                rows.append(row)

            if not rows or not self._exprs:
                # Global shuffle: no hierarchy, just randomize order
                self._rng.shuffle(rows)
                self._shuffled_rows = rows
            else:
                # If expressions are specified, shuffle the rows hierarchically
                # by the expressions. For expression (a, b, c):
                # 1. Shuffle (a) groups globally
                # 2. Shuffle (a, b) groups within each (a) group
                # 3. Shuffle (a, b, c) groups within each (a, b) group
                # 4. Shuffle rows within each (a, b, c) group if shuffle_rows is True
                input_schema = self._input.schema()
                # Precompute all IDs for each row to avoid repeated evaluation
                ids = [self._get_id(row, input_schema, self._exprs) for row in rows]
                # Shuffle the indices of the rows based on the IDs
                order = self._shuffle_level(list(range(len(rows))), 0, ids)
                self._shuffled_rows = [rows[i] for i in order]

        while self._index < len(self._shuffled_rows):
            row = self._shuffled_rows[self._index]
            self._index += 1
            if not self.should_skip(row, self._input.schema()):
                return row

        return None

    def _shuffle_level(
        self,
        indices: list[int],
        depth: int,
        ids: list[tuple[object, ...]],
    ) -> list[int]:
        """Recursively shuffle indices at each hierarchy level."""

        if depth == len(self._exprs):
            if self._shuffle_rows:
                self._rng.shuffle(indices)
            return indices

        def key_fn(i: int) -> tuple[object, ...]:
            return ids[i][: depth + 1]

        groups = [
            self._shuffle_level(list(g), depth + 1, ids)
            for _, g in groupby(sorted(indices, key=key_fn), key=key_fn)
        ]
        self._rng.shuffle(groups)
        return [i for group in groups for i in group]

    def close(self) -> None:
        self._input.close()


class Log(PhysicalPlan):
    def __init__(self, path: str, input: PhysicalPlan, schema: Schema) -> None:
        super().__init__()
        self._path = path
        self._input = input
        self._schema = schema
        self._rows: list[Row] = []

    def inputs(self) -> tuple[PhysicalPlan, ...]:
        return (self._input,)

    def schema(self) -> Schema:
        return self._schema

    def open(self) -> None:
        self._input.open()

    def next(self) -> Row | None:
        while (row := self._input.next()) is not None:
            if self.should_skip(row, self._input.schema()):
                continue
            self._rows.append(row)
            return row

        return None

    def close(self) -> None:
        with open(self._path, "wb") as f:
            pickle.dump((self._rows, self._schema), f)
        logger.debug("Logged %d rows to %s", len(self._rows), self._path)
        self._input.close()
