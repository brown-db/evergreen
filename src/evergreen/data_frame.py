from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from evergreen.catalog.schema import Schema
from evergreen.core.session_state import SessionState
from evergreen.model.embedding_model import EmbeddingModelMetrics
from evergreen.model.language_model import LanguageModelMetrics
from evergreen.planner.logical.expr import Expr, col
from evergreen.planner.logical.plan import LogicalPlan
from evergreen.planner.logical.plan_builder import LogicalPlanBuilder
from evergreen.planner.physical.planner import PhysicalPlanner
from evergreen.provenance import Token
from evergreen.storage.row import Row

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryMetrics:
    planning_latency: float
    execution_latency: float
    language_model_metrics: tuple[LanguageModelMetrics, ...]
    embedding_model_metrics: tuple[EmbeddingModelMetrics, ...]
    optimizer_language_model_metrics: tuple[LanguageModelMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "planning_latency": self.planning_latency,
            "execution_latency": self.execution_latency,
            "language_model_metrics": [
                metric.to_dict() for metric in self.language_model_metrics
            ],
            "embedding_model_metrics": [
                metric.to_dict() for metric in self.embedding_model_metrics
            ],
            "optimizer_language_model_metrics": [
                metric.to_dict() for metric in self.optimizer_language_model_metrics
            ],
        }


@dataclass(frozen=True)
class QueryResult:
    rows: list[Row]
    metrics: QueryMetrics


@dataclass(frozen=True)
class FilterMetrics:
    tp: int  # Number of true positives
    fp: int  # Number of false positives
    fn: int  # Number of false negatives

    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    @property
    def f1_score(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    def to_dict(self) -> dict[str, object]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
        }


@dataclass(frozen=True)
class MapMetrics:
    correct_by_column: dict[str, int]
    total_rows: int

    @property
    def accuracy(self) -> float:
        if self.total_rows == 0 or not self.correct_by_column:
            return 0.0
        total_correct = sum(self.correct_by_column.values())
        total_comparisons = self.total_rows * len(self.correct_by_column)
        return total_correct / total_comparisons

    def to_dict(self) -> dict[str, object]:
        return {
            "correct_by_column": self.correct_by_column,
            "total_rows": self.total_rows,
            "accuracy": self.accuracy,
        }


class DataFrame:
    """A lazy query builder for semantic queries over structured and unstructured data.

    `DataFrame` operations are chainable and lazily evaluated. Build queries
    by chaining `filter()`, `map()`, `aggregate()`, `with_rank()`, `check()`, etc.
    """

    def __init__(self, plan: LogicalPlan, session_state: SessionState) -> None:
        self._plan = plan
        self._session_state = session_state

    def logical_plan(self) -> LogicalPlan:
        return self._plan

    def schema(self) -> Schema:
        return self._plan.schema()

    def filter(self, predicate: Expr) -> DataFrame:
        """Filter rows based on a predicate expression.

        Returns a new `DataFrame` containing only rows where the predicate
        evaluates to `True`.

        Args:
            predicate: A boolean expression to evaluate for each row.
                Rows where the predicate is `True` are included in the result.

        Returns:
            A new `DataFrame` with filtered rows.

        Examples:
            Filter reviews with 5-star ratings:

            >>> df.filter(col("stars").eq(5))

            Filter using a prompt expression:

            >>> df.filter(
            ...     prompt("The movie {review} describes the movie's cinematography")
            ... )
        """
        new_plan = LogicalPlanBuilder(self._plan).filter(predicate).build()
        return DataFrame(new_plan, self._session_state)

    def map(self, expr: Expr) -> DataFrame:
        """Add a new column by evaluating an expression.

        Projects all existing columns plus a new column computed from the
        given expression. The new column name is derived from the expression
        unless an alias is provided.

        Args:
            expr: An expression to evaluate for each row.
                Use `Expr.alias(name)` to specify a custom column name.

        Returns:
            A new `DataFrame` with all existing columns plus the new computed column.

        Examples:
            Add a semantic classification column:

            >>> class Sentiment(Enum):
            ...     POSITIVE = "positive"
            ...     NEGATIVE = "negative"
            ...     MIXED = "mixed"
            ...     NEUTRAL = "neutral"
            >>> df.map(
            ...     prompt(
            ...         "Identify the sentiment of the movie {review} towards the "
            ...         "movie",
            ...         Sentiment
            ...     ).alias("sentiment")
            ... )

            Add a boolean flag column:

            >>> df.map(
            ...     prompt(
            ...         "Identify whether the movie {review} says that the movie's "
            ...         "ticket price is high",
            ...         bool
            ...     ).alias("price_is_high")
            ... )
        """
        exprs = [col(name) for name in self._plan.schema().field_names()]
        exprs.append(expr)
        new_plan = LogicalPlanBuilder(self._plan).project(exprs).build()
        return DataFrame(new_plan, self._session_state)

    def aggregate(
        self, agg_exprs: Sequence[Expr], group_by: Sequence[Expr] = ()
    ) -> DataFrame:
        """Compute aggregate values over the `DataFrame`.

        Applies aggregate functions to compute summary statistics. Optionally
        groups rows by one or more expressions before aggregation.

        Args:
            agg_exprs: A sequence of aggregate expressions to compute.
                Common aggregates include `count_if()`, `proportion()`,
                `bool_and()`, and `bool_or()`.
            group_by: Optional sequence of expressions to group by before
                aggregating. Defaults to no grouping (aggregate over all rows).

        Returns:
            A new `DataFrame` with aggregated results. Without grouping, returns
            a single row. With grouping, returns one row per unique group.

        Examples:
            Count reviews mentioning violence:

            >>> (
            ...     df.map(
            ...         prompt(
            ...             "Identify whether the movie {review} says that the movie "
            ...             "has too much violence", bool
            ...         ).alias("has_too_much_violence")
            ...     )
            ...     .aggregate(
            ...         [count_if(col("has_too_much_violence")).alias("violence_count")]
            ...     )
            ... )

            Among the reviews that discuss the movie's cinematography,
            compute the proportion of reviews that call the cinematography beautiful:

            >>> (
            ...     df.filter(
            ...         prompt("The movie {review} mentions the movie's cinematography")
            ...     )
            ...     .map(
            ...         prompt(
            ...             "Identify whether the movie {review} calls the movie's "
            ...             "cinematography beautiful",
            ...             bool
            ...         ).alias("is_beautiful")
            ...     )
            ...     .aggregate(
            ...         [proportion(col("is_beautiful")).alias("beautiful_prop")]
            ...     )
            ... )

            Group by star rating and compute the proportion of reviews mentioning
            the movie's pacing:

            >>> (
            ...     df.map(
            ...         prompt(
            ...             "Identify whether the movie {review} mentions the movie's "
            ...             "pacing",
            ...             bool
            ...         ).alias("mentions_pacing")
            ...     )
            ...     .aggregate(
            ...         [proportion(col("mentions_pacing")).alias("pacing_prop")],
            ...         group_by=[col("stars")]
            ...     )
            ... )
        """
        new_plan = LogicalPlanBuilder(self._plan).aggregate(agg_exprs, group_by).build()
        return DataFrame(new_plan, self._session_state)

    def with_rank(self, expr: Expr, descending: bool = True) -> DataFrame:
        """Add a rank column based on an expression.

        Computes a ranking over rows based on the specified expression and
        adds it as a new column.

        Args:
            expr: The expression to rank by.
            descending: If True (default), higher values receive lower ranks
                (rank 1 = highest value). If False, lower values receive
                lower ranks.

        Returns:
            A new `DataFrame` with an additional rank column.

        Examples:
            Rank individual rows by a column:

            >>> df.with_rank(col("stars"))  # Highest first (default)
            >>> df.with_rank(col("stars"), descending=False)  # Lowest first

            Rank aggregated groups to compare which group has the highest metric:

            >>> (
            ...     df.aggregate(
            ...         [proportion(col("is_recommended")).alias("rec_rate")],
            ...         group_by=[col("category")]
            ...     )
            ...     .with_rank(col("rec_rate"))
            ...     .filter(col("category").eq("electronics"))
            ...     .check(col("rank").eq(1))
            ... )
        """
        new_plan = LogicalPlanBuilder(self._plan).with_rank(expr, descending).build()
        return DataFrame(new_plan, self._session_state)

    def check(self, predicate: Expr) -> DataFrame:
        """Add a check column based on a predicate.

        Adds a boolean column indicating whether each row satisfies the
        given predicate. The name of the new column is derived from the
        predicate.

        Args:
            predicate: A boolean expression to check for each row.

        Returns:
            A new `DataFrame` with a boolean check column.

        Examples:
            Check that all reviews are verified:

            >>> df.check(col("all_verified"))

            Check that the count is greater than or equal to 2:

            >>> df.check(col("count") >= 2)
        """
        new_plan = LogicalPlanBuilder(self._plan).check(predicate).build()
        return DataFrame(new_plan, self._session_state)

    def log(self, path: str) -> DataFrame:
        new_plan = LogicalPlanBuilder(self._plan).log(path).build()
        return DataFrame(new_plan, self._session_state)

    def collect(self) -> QueryResult:
        self._session_state.clear_language_model_metrics()
        self._session_state.clear_embedding_model_metrics()
        self._session_state.clear_optimizer_language_model_metrics()

        t0 = time.perf_counter()
        logical_plan = self._session_state.optimize_logical_plan(self._plan)
        physical_plan = self._session_state.create_physical_plan(logical_plan)

        planning_latency = time.perf_counter() - t0
        optimizer_language_model_metrics = (
            self._session_state.optimizer_language_model_metrics()
        )

        displayed_plans = self._display_plans(self._plan, logical_plan)
        logger.debug("explain query:\n%s", displayed_plans)

        t0 = time.perf_counter()
        physical_plan.open()
        try:
            rows: list[Row] = []
            while (row := physical_plan.next()) is not None:
                rows.append(row)

            execution_latency = time.perf_counter() - t0
            language_model_metrics = self._session_state.language_model_metrics()
            embedding_model_metrics = self._session_state.embedding_model_metrics()
            query_metrics = QueryMetrics(
                planning_latency,
                execution_latency,
                language_model_metrics,
                embedding_model_metrics,
                optimizer_language_model_metrics,
            )
            query_result = QueryResult(rows, query_metrics)

            return query_result
        finally:
            physical_plan.close()

    def explain(self) -> str:
        optimized_logical_plan = self._session_state.optimize_logical_plan(self._plan)
        return self._display_plans(self._plan, optimized_logical_plan)

    def optimized_logical_plan(self) -> LogicalPlan:
        return self._session_state.optimize_logical_plan(self._plan)

    def evaluate_tokens(self, tokens: Iterable[Token]) -> dict[Token, bool]:
        rows = self.collect().rows
        schema = self.schema()
        index = {row.id(schema): row for row in rows}

        results: dict[Token, bool] = {}

        for token in tokens:
            row = index.get(token.row_id)

            if row is None:
                results[token] = False
            else:
                physical_predicate = PhysicalPlanner.create_expr(
                    token.predicate, schema, self._session_state
                )
                value = physical_predicate.evaluate(row, token.row_id).value

                if not isinstance(value, bool):
                    raise ValueError(f"Predicate must return bool, got {type(value)}")

                is_valid = value == token.sign
                results[token] = is_valid

        return results

    def count_valid_tokens(self, tokens: Iterable[Token]) -> int:
        return sum(self.evaluate_tokens(tokens).values())

    def evaluate_filter(
        self, pre_sem_op_df: DataFrame, post_sem_op_df: DataFrame
    ) -> FilterMetrics:
        post_sem_op_reference_schema = self.schema()
        pre_sem_op_schema = pre_sem_op_df.schema()
        post_sem_op_schema = post_sem_op_df.schema()

        post_sem_op_reference_row_ids = {
            row.id(post_sem_op_reference_schema) for row in self.collect().rows
        }
        pre_sem_op_row_ids = {
            row.id(pre_sem_op_schema) for row in pre_sem_op_df.collect().rows
        }
        post_sem_op_row_ids = {
            row.id(post_sem_op_schema) for row in post_sem_op_df.collect().rows
        }

        # TP: rows correct selected
        # (in both reference and implementation output)
        tp = len(post_sem_op_reference_row_ids & post_sem_op_row_ids)

        # FP: rows incorrect selected
        # (in implementation output but not reference output)
        fp = len(post_sem_op_row_ids - post_sem_op_reference_row_ids)

        # FN: rows incorrectly not selected
        # (in reference output and in implementation input,
        # but not in implementation output)
        fn = len(
            (post_sem_op_reference_row_ids & pre_sem_op_row_ids) - post_sem_op_row_ids
        )

        return FilterMetrics(tp, fp, fn)

    def evaluate_map(
        self, post_sem_op_df: DataFrame, semantic_map_columns: Sequence[Expr]
    ) -> MapMetrics:
        post_sem_op_reference_schema = self.schema()
        post_sem_op_schema = post_sem_op_df.schema()

        post_sem_op_reference_rows = {
            row.id(post_sem_op_reference_schema): row for row in self.collect().rows
        }
        post_sem_op_rows = {
            row.id(post_sem_op_schema): row for row in post_sem_op_df.collect().rows
        }

        common_row_ids = post_sem_op_reference_rows.keys() & post_sem_op_rows.keys()

        correct_by_column: dict[str, int] = {}

        for expr in semantic_map_columns:
            col_name = str(expr)
            reference_index = post_sem_op_reference_schema.index_of(col_name)
            index = post_sem_op_schema.index_of(col_name)

            correct = sum(
                1
                for row_id in common_row_ids
                if post_sem_op_reference_rows[row_id][reference_index]
                == post_sem_op_rows[row_id][index]
            )
            correct_by_column[col_name] = correct

        return MapMetrics(correct_by_column, len(common_row_ids))

    @classmethod
    def _display_plans(
        cls, initial_plan: LogicalPlan, optimized_plan: LogicalPlan
    ) -> str:
        sections = [
            "Explain:",
            "",
            "Initial Logical Plan:",
            initial_plan.display(),
            "",
            "Optimized Logical Plan:",
            optimized_plan.display(),
        ]
        return "\n".join(sections)
