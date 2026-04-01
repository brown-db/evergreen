from __future__ import annotations

from collections.abc import Sequence

from evergreen.catalog.table import TableProvider
from evergreen.planner.logical.expr import Expr
from evergreen.planner.logical.plan import (
    Aggregate,
    Check,
    Filter,
    Log,
    LogicalPlan,
    Projection,
    TableScan,
    WithRank,
)


class LogicalPlanBuilder:
    def __init__(self, plan: LogicalPlan) -> None:
        self._plan = plan

    @classmethod
    def table_scan(cls, table_provider: TableProvider) -> LogicalPlanBuilder:
        return cls(TableScan(table_provider))

    def filter(self, predicate: Expr) -> LogicalPlanBuilder:
        return LogicalPlanBuilder(Filter(predicate, self._plan))

    def check(self, predicate: Expr) -> LogicalPlanBuilder:
        return LogicalPlanBuilder(Check(predicate, self._plan))

    def project(self, exprs: Sequence[Expr]) -> LogicalPlanBuilder:
        return LogicalPlanBuilder(Projection(tuple(exprs), self._plan))

    def aggregate(
        self, agg_exprs: Sequence[Expr], group_exprs: Sequence[Expr]
    ) -> LogicalPlanBuilder:
        return LogicalPlanBuilder(
            Aggregate(
                tuple(agg_exprs),
                tuple(group_exprs),
                (None,) * len(agg_exprs),
                self._plan,
            )
        )

    def with_rank(self, expr: Expr, descending: bool) -> LogicalPlanBuilder:
        return LogicalPlanBuilder(WithRank(expr, descending, self._plan))

    def log(self, path: str) -> LogicalPlanBuilder:
        return LogicalPlanBuilder(Log(path, self._plan))

    def build(self) -> LogicalPlan:
        return self._plan
