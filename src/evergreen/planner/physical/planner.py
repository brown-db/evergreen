from __future__ import annotations

import re
from typing import TYPE_CHECKING

from evergreen.catalog.schema import Schema
from evergreen.common.constants import FIELD_NAME_REGEX
from evergreen.planner.logical import expr as logi_expr
from evergreen.planner.logical import plan as logi_plan
from evergreen.planner.logical.expr import Expr
from evergreen.planner.logical.plan import LogicalPlan
from evergreen.planner.physical import expr as phys_expr
from evergreen.planner.physical import plan as phys_plan
from evergreen.planner.physical.expr import AggregateFunctionExpr, PhysicalExpr
from evergreen.planner.physical.plan import (
    CountProvider,
    CountScan,
    Filter,
    FusedFilterProjection,
    PhysicalPlan,
    SkipPredicate,
    Sort,
)

if TYPE_CHECKING:
    from evergreen.core.session_state import SessionState


class PhysicalPlanner:
    def create_physical_plan(
        self, logical_plan: LogicalPlan, session_state: SessionState
    ) -> PhysicalPlan:
        plan = self._create_plan(logical_plan, session_state)
        self._set_skip_predicates(plan)
        self._set_count_provider(plan)
        self._set_confidence_levels(plan, session_state.confidence_level())
        return plan

    @classmethod
    def _set_skip_predicates(cls, plan: PhysicalPlan) -> None:
        match plan:
            case phys_plan.StreamAggregate(_, group_exprs, _, early_stop_enabled) if (
                group_exprs and early_stop_enabled
            ):
                input = plan.inputs()[0]
                cls._propagate_skip_predicate(plan.in_skip_group, input)
            case _:
                pass

        for input in plan.inputs():
            cls._set_skip_predicates(input)

    @classmethod
    def _propagate_skip_predicate(
        cls, predicate: SkipPredicate, plan: PhysicalPlan
    ) -> None:
        plan.add_skip_predicate(predicate)
        for input in plan.inputs():
            cls._propagate_skip_predicate(predicate, input)

    @classmethod
    def _set_count_provider(cls, plan: PhysicalPlan) -> None:
        """Sets count providers for aggregates to enable precomputed totals.

        For nested aggregates, we need to tell the outer aggregate how many
        GROUP BY columns the inner aggregate has.
        """
        match plan:
            case phys_plan.StreamAggregate():
                input = plan.inputs()[0]
                count_provider = cls._find_count_provider(input)
                if count_provider is not None:
                    # Determine if input is a grouped aggregate
                    # If so, pass its group key length so we count groups
                    match input:
                        case phys_plan.StreamAggregate(group_exprs):
                            input_group_key_length = len(group_exprs)
                        case _:
                            input_group_key_length = 0
                    plan.set_count_provider(count_provider, input_group_key_length)
            case _:
                pass

        for input in plan.inputs():
            cls._set_count_provider(input)

    @classmethod
    def _find_count_provider(cls, plan: PhysicalPlan) -> CountProvider | None:
        match plan:
            # TODO: Since we currently don't allow users to sort via the
            # DataFrame API, this implementation is fine. However, this hack
            # will break if we allow users to sort via the DataFrame API, since
            # their inserted Sort may be incorrectly considered as a count
            # provider.
            case Sort() | CountScan():
                return plan
            case Filter() | FusedFilterProjection():
                # If we see a Filter or FusedFilterProjection and haven't seen a
                # Sort yet, this means that there is no valid count provider,
                # since they don't provide correct counts.
                return None
            case _:
                inputs = plan.inputs()
                if inputs:
                    return cls._find_count_provider(inputs[0])
                return None

    @classmethod
    def _set_confidence_levels(
        cls, plan: PhysicalPlan, confidence_level: float | None
    ) -> None:
        if confidence_level is None:
            return

        count = cls._count_aggregate_operators_supporting_estimation(plan)
        if count == 0:
            return

        # Split confidence level equally among the aggregate operators that
        # support estimation
        alpha = 1 - confidence_level
        adjusted_alpha = alpha / count
        adjusted_confidence_level = 1 - adjusted_alpha

        cls._propagate_confidence_level(plan, adjusted_confidence_level)

    @classmethod
    def _count_aggregate_operators_supporting_estimation(
        cls, plan: PhysicalPlan
    ) -> int:
        count = 0

        match plan:
            case phys_plan.StreamAggregate(agg_exprs):
                is_above_shuffle = plan.is_above(phys_plan.Shuffle)
                is_above_relevance_sort = plan.is_above(phys_plan.RelevanceSort)
                is_above_stream_aggregate = plan.is_above(phys_plan.StreamAggregate)

                # Determine whether the aggregate has an associated
                # Shuffle and RelevanceSort
                # Inner aggregate uses RelevanceSort; outer aggregate uses Shuffle
                has_shuffle = is_above_shuffle and (
                    not is_above_relevance_sort or is_above_stream_aggregate
                )

                for expr in agg_exprs:
                    expr.has_shuffle = has_shuffle

                if any(expr.supports_estimation() for expr in agg_exprs):
                    count = 1
            case _:
                pass

        return count + sum(
            cls._count_aggregate_operators_supporting_estimation(input)
            for input in plan.inputs()
        )

    @classmethod
    def _propagate_confidence_level(
        cls, plan: PhysicalPlan, confidence_level: float
    ) -> None:
        match plan:
            case phys_plan.StreamAggregate(agg_exprs):
                if any(expr.supports_estimation() for expr in agg_exprs):
                    plan.set_confidence_level(confidence_level)
            case _:
                pass

        for input in plan.inputs():
            cls._propagate_confidence_level(input, confidence_level)

    @classmethod
    def _create_plan(
        cls, logical_plan: LogicalPlan, session_state: SessionState
    ) -> PhysicalPlan:
        match logical_plan:
            case logi_plan.TableScan(table_provider):
                return table_provider.scan()
            case logi_plan.Filter(predicate, input):
                return phys_plan.Filter(
                    cls.create_expr(predicate, input.schema(), session_state),
                    cls._create_plan(input, session_state),
                    logical_plan.schema(),
                    session_state.batch_size(),
                )
            case logi_plan.Check(predicate, input):
                return phys_plan.Check(
                    cls.create_expr(predicate, input.schema(), session_state),
                    cls._create_plan(input, session_state),
                    logical_plan.schema(),
                )
            case logi_plan.Projection(exprs, input):
                return phys_plan.Projection(
                    tuple(
                        cls.create_expr(expr, input.schema(), session_state)
                        for expr in exprs
                    ),
                    cls._create_plan(input, session_state),
                    logical_plan.schema(),
                    session_state.batch_size(),
                )
            case logi_plan.FusedFilterProjection(exprs, plan_types, input):
                return phys_plan.FusedFilterProjection(
                    tuple(
                        cls.create_expr(expr, input.schema(), session_state)
                        for expr in exprs
                    ),
                    plan_types,
                    cls._create_plan(input, session_state),
                    logical_plan.schema(),
                    session_state.batch_size(),
                )
            case logi_plan.Sort(exprs, reverse, input):
                return phys_plan.Sort(
                    tuple(
                        cls.create_expr(expr, input.schema(), session_state)
                        for expr in exprs
                    ),
                    reverse,
                    cls._create_plan(input, session_state),
                    logical_plan.schema(),
                )
            case logi_plan.RelevanceSort(
                text_column_name,
                _,
                query_embedding,
                inclusion_keywords,
                exclusion_keywords,
                input,
            ):
                return phys_plan.RelevanceSort(
                    text_column_name,
                    query_embedding,
                    inclusion_keywords,
                    exclusion_keywords,
                    cls._create_plan(input, session_state),
                    logical_plan.schema(),
                )
            case logi_plan.WithRank(expr, descending, input):
                return phys_plan.WithRank(
                    cls.create_expr(expr, input.schema(), session_state),
                    descending,
                    cls._create_plan(input, session_state),
                    logical_plan.schema(),
                    session_state.minimal_provenance(),
                )
            case logi_plan.Aggregate(
                agg_exprs, group_exprs, early_stop_comparisons, input
            ):
                return phys_plan.StreamAggregate(
                    tuple(
                        cls._create_aggregate_expr(
                            agg_expr, input.schema(), session_state
                        )
                        for agg_expr in agg_exprs
                    ),
                    tuple(
                        cls.create_expr(group_expr, input.schema(), session_state)
                        for group_expr in group_exprs
                    ),
                    early_stop_comparisons,
                    session_state.early_stop_enabled(),
                    cls._create_plan(input, session_state),
                    logical_plan.schema(),
                )
            case logi_plan.CountScan(input):
                return phys_plan.CountScan(
                    cls._create_plan(input, session_state),
                    logical_plan.schema(),
                )
            case logi_plan.Shuffle(exprs, shuffle_rows, seed, input):
                return phys_plan.Shuffle(
                    tuple(
                        cls.create_expr(expr, input.schema(), session_state)
                        for expr in exprs
                    ),
                    shuffle_rows,
                    seed,
                    cls._create_plan(input, session_state),
                    logical_plan.schema(),
                )
            case logi_plan.Log(path, input):
                return phys_plan.Log(
                    path,
                    cls._create_plan(input, session_state),
                    logical_plan.schema(),
                )
            case _:
                raise ValueError(f"Unsupported logical plan: {logical_plan}")

    @classmethod
    def create_expr(
        cls, logical_expr: Expr, input_schema: Schema, session_state: SessionState
    ) -> PhysicalExpr:
        match logical_expr:
            case logi_expr.Alias(expr, _):
                return cls.create_expr(expr, input_schema, session_state)
            case logi_expr.Column(name):
                return phys_expr.Column(name, input_schema.index_of(name))
            case logi_expr.Literal(value):
                return phys_expr.Literal(value)
            case logi_expr.BinaryExpr(left, op, right):
                return phys_expr.BinaryExpr(
                    cls.create_expr(left, input_schema, session_state),
                    op,
                    cls.create_expr(right, input_schema, session_state),
                    session_state.minimal_provenance(),
                )
            case logi_expr.Not(expr):
                return phys_expr.Not(
                    cls.create_expr(expr, input_schema, session_state),
                )
            case logi_expr.Prompt(prompt_str, return_type):
                field_names = re.findall(FIELD_NAME_REGEX, prompt_str)
                field_indices = tuple(
                    (field_name, input_schema.index_of(field_name))
                    for field_name in field_names
                )
                return phys_expr.Prompt(
                    prompt_str,
                    return_type,
                    field_indices,
                    session_state.language_model(),
                )
            case _:
                raise ValueError(f"Unsupported logical expression: {logical_expr}")

    @classmethod
    def _create_aggregate_expr(
        cls, logical_expr: Expr, input_schema: Schema, session_state: SessionState
    ) -> AggregateFunctionExpr:
        match logical_expr:
            case logi_expr.Alias(expr, _):
                return cls._create_aggregate_expr(expr, input_schema, session_state)
            case logi_expr.AggregateFunction(udaf, args):
                return AggregateFunctionExpr(
                    udaf,
                    tuple(
                        cls.create_expr(arg, input_schema, session_state)
                        for arg in args
                    ),
                    session_state.relative_error(),
                    session_state.minimal_provenance(),
                )
            case _:
                raise ValueError(f"Unsupported aggregate expression: {logical_expr}")
