from __future__ import annotations

import reprlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

from evergreen.catalog.schema import Field, Schema
from evergreen.planner.logical.types import Operator
from evergreen.planner.logical.udaf import (
    AggregateUDF,
    BoolAndUDF,
    BoolOrUDF,
    CountIfUDF,
    ProportionUDF,
)


class Expr(ABC):
    """Base class for expressions in semantic queries.

    Expressions can be combined using comparison operators (`<`, `<=`, `>`, `>=`),
    logical operators (`&`, `|`, `~`), and equality methods (`eq()`, `ne()`).
    Use `alias()` to name computed columns.
    """

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def to_field(self, input_schema: Schema) -> Field:
        pass

    def alias(self, name: str) -> Alias:
        """Assign a custom name to this expression.

        Creates an aliased version of the expression with the specified name.
        Useful for naming computed columns in operations.

        Args:
            name: The alias name to assign.

        Returns:
            An aliased expression with the given name.

        Examples:
            Name a computed column:

            >>> prompt("Identify the sentiment of the {review}", str).alias("sentiment")
        """
        return Alias(self, name)

    def eq(self, other: object) -> BinaryExpr:
        """Test equality with another expression or value.

        Note: Use `eq()` instead of `==` because Python's `==` operator
        cannot be overloaded to return an expression.

        Args:
            other: The value or expression to compare with.

        Returns:
            A binary expression representing the equality test.

        Examples:
            >>> col("status").eq("active")
            >>> col("category").eq(col("default_category"))
        """
        if not isinstance(other, Expr):
            other = lit(other)
        return BinaryExpr(self, Operator.EQ, other)

    def ne(self, other: object) -> BinaryExpr:
        """Test inequality with another expression or value.

        Note: Use `ne()` instead of `!=` because Python's `!=` operator
        cannot be overloaded to return an expression.

        Args:
            other: The value or expression to compare with.

        Returns:
            A binary expression representing the inequality test.

        Examples:
            >>> col("status").ne("deleted")
        """
        if not isinstance(other, Expr):
            other = lit(other)
        return BinaryExpr(self, Operator.NE, other)

    def __lt__(self, other: object) -> BinaryExpr:
        """Less than comparison (`<`).

        Examples:
            >>> col("stars") < 3
        """
        if not isinstance(other, Expr):
            other = lit(other)
        return BinaryExpr(self, Operator.LT, other)

    def __le__(self, other: object) -> BinaryExpr:
        """Less than or equal comparison (`<=`).

        Examples:
            >>> col("stars") <= 3
        """
        if not isinstance(other, Expr):
            other = lit(other)
        return BinaryExpr(self, Operator.LE, other)

    def __gt__(self, other: object) -> BinaryExpr:
        """Greater than comparison (`>`).

        Examples:
            >>> col("stars") > 3
        """
        if not isinstance(other, Expr):
            other = lit(other)
        return BinaryExpr(self, Operator.GT, other)

    def __ge__(self, other: object) -> BinaryExpr:
        """Greater than or equal comparison (`>=`).

        Examples:
            >>> col("stars") >= 3
        """
        if not isinstance(other, Expr):
            other = lit(other)
        return BinaryExpr(self, Operator.GE, other)

    def __and__(self, other: object) -> BinaryExpr:
        """Logical AND (`&`).

        Combine two boolean expressions. Use parentheses due to Python's
        operator precedence.

        Examples:
            >>> (col("stars") >= 4) & col("verified")
        """
        if not isinstance(other, Expr):
            other = lit(other)
        return BinaryExpr(self, Operator.AND, other)

    def __or__(self, other: object) -> BinaryExpr:
        """Logical OR (`|`).

        Combine two boolean expressions. Use parentheses due to Python's
        operator precedence.

        Examples:
            >>> (col("stars").eq(5)) | col("featured")
        """
        if not isinstance(other, Expr):
            other = lit(other)
        return BinaryExpr(self, Operator.OR, other)

    def __invert__(self) -> Not:
        """Logical NOT (`~`).

        Negate a boolean expression.

        Examples:
            >>> ~col("is_spam")
        """
        return Not(self)

    __rand__ = __and__
    __ror__ = __or__


@dataclass(frozen=True)
class Alias(Expr):
    expr: Expr
    name: str

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.expr!r}.alias({self.name})"

    def to_field(self, input_schema: Schema) -> Field:
        inner = self.expr.to_field(input_schema)
        return Field(str(self), inner.dtype)


@dataclass(frozen=True)
class Column(Expr):
    name: str

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"col({self.name})"

    def to_field(self, input_schema: Schema) -> Field:
        index = input_schema.index_of(str(self))
        return input_schema.fields[index]


def col(name: str) -> Expr:
    """Create a column reference expression.

    References an existing column in the `DataFrame` by name. Used to
    access column values in filter predicates, aggregations, and other
    expressions.

    Args:
        name: The name of the column to reference.

    Returns:
        An expression representing the column reference.

    Examples:
        Reference a column in a filter:

        >>> df.filter(col("stars") >= 4)

        Use in grouping and aggregation:

        >>> df.aggregate(
        ...     [count_if(col("mentions_violence")).alias("mentions_violence_count")],
        ...     group_by=[col("category")]
        ... )
    """
    return Column(name)


@dataclass(frozen=True)
class Literal(Expr):
    value: object

    def __str__(self) -> str:
        return repr(self.value)

    def __repr__(self) -> str:
        return f"lit({reprlib.repr(self.value)})"

    def to_field(self, input_schema: Schema) -> Field:
        return Field(str(self), type(self.value))


def lit(value: object) -> Expr:
    """Create a literal value expression.

    Wraps a Python value as an expression for use in comparisons and
    other operations. Literal conversion is automatic in most cases,
    but explicit use can improve clarity.

    Args:
        value: The literal value (string, number, bool, list, etc.).

    Returns:
        An expression representing the literal value.

    Examples:
        Explicit literal in comparison:

        >>> col("status").eq(lit("active"))
    """
    return Literal(value)


@dataclass(frozen=True)
class BinaryExpr(Expr):
    left: Expr
    op: Operator
    right: Expr

    def __str__(self) -> str:
        return f"{self.left} {self.op.value} {self.right}"

    def __repr__(self) -> str:
        return f"{self.left!r} {self.op.value} {self.right!r}"

    def to_field(self, input_schema: Schema) -> Field:
        dtype = self.op.return_type()
        return Field(str(self), dtype)


@dataclass(frozen=True)
class Not(Expr):
    expr: Expr

    def __str__(self) -> str:
        return f"~{self.expr}"

    def __repr__(self) -> str:
        return f"~{self.expr!r}"

    def to_field(self, input_schema: Schema) -> Field:
        return Field(str(self), bool)


@dataclass(frozen=True)
class Prompt(Expr):
    prompt_str: str
    return_type: type[object]

    def __str__(self) -> str:
        return f"prompt({self.prompt_str!r})"

    def __repr__(self) -> str:
        return str(self)

    def to_field(self, input_schema: Schema) -> Field:
        return Field(str(self), self.return_type)


def prompt(prompt_str: str, return_type: type[object] = bool) -> Prompt:
    """Create a prompt expression.

    Evaluates a natural language prompt against each row using a language
    model. The prompt can reference column values using template syntax
    and returns the specified type.

    Args:
        prompt_str: A natural language prompt describing the predicate
            or function to evaluate. Use `{column_name}` to reference column values.
        return_type: The expected return type of the prompt evaluation.
            Defaults to `bool`, making the prompt a predicate that returns
            `True` or `False`. Other types include `str`, `Enum`, etc.

    Returns:
        A prompt expression that will be evaluated by the language model.

    Examples:
        Boolean predicate prompt:

        >>> df.filter(prompt("The {review} mentions the movie's cinematography"))

        Extraction prompt:

        >>> df.map(prompt("Extract the main topic of the {review}", str).alias("topic"))

        Classification prompt:

        >>> class Sentiment(Enum):
        ...     POSITIVE = "positive"
        ...     NEGATIVE = "negative"
        ...     MIXED = "mixed"
        ...     NEUTRAL = "neutral"
        >>> df.map(
        ...     prompt(
        ...         "Identify the sentiment of the {review}", Sentiment
        ...     ).alias("sentiment")
        ... )
    """
    return Prompt(prompt_str, return_type)


@dataclass(frozen=True)
class AggregateFunction(Expr):
    udaf: AggregateUDF
    args: tuple[Expr, ...]

    def __str__(self) -> str:
        return f"{self.udaf.name()}({', '.join(str(arg) for arg in self.args)})"

    def __repr__(self) -> str:
        return f"{self.udaf.name()}({', '.join(repr(arg) for arg in self.args)})"

    def to_field(self, input_schema: Schema) -> Field:
        arg_types = tuple(arg.to_field(input_schema).dtype for arg in self.args)
        return Field(str(self), self.udaf.return_type(arg_types))


def count_if(expr: Expr) -> AggregateFunction:
    """Count rows where an expression evaluates to `True`.

    An aggregate function that counts the number of rows for which the
    given boolean expression is `True`.

    Args:
        expr: A boolean expression to evaluate for each row.

    Returns:
        An aggregate expression that computes the count.

    Examples:
        Count positive reviews:

        >>> df.aggregate([count_if(col("is_positive"))])

        Count high-rated items:

        >>> df.aggregate([count_if(col("stars") >= 4)])
    """
    return AggregateFunction(CountIfUDF(), (expr,))


def proportion(expr: Expr) -> AggregateFunction:
    """Compute the proportion of rows where an expression is `True`.

    An aggregate function that calculates the ratio of rows where the
    given boolean expression is `True` to the total number of rows.
    Returns a value between `0.0` and `1.0`.

    Args:
        expr: A boolean expression to evaluate for each row.

    Returns:
        An aggregate expression that computes the proportion.

    Examples:
        Proportion of positive reviews:

        >>> df.aggregate([proportion(col("is_positive"))])

        Proportion by category:

        >>> df.aggregate(
        ...     [proportion(col("mentions_violence"))],
        ...     group_by=[col("category")]
        ... )
    """
    return AggregateFunction(ProportionUDF(), (expr,))


def bool_and(expr: Expr) -> AggregateFunction:
    """Compute logical AND across all rows.

    An aggregate function that returns `True` only if the expression
    evaluates to `True` for all rows. Returns `False` if any row evaluates
    to `False`. In other words, this is a universal quantifier.

    Args:
        expr: A boolean expression to evaluate for each row.

    Returns:
        An aggregate expression that computes the logical AND.

    Examples:
        Check if all reviews are verified:

        >>> df.aggregate([bool_and(col("verified"))])
    """
    return AggregateFunction(BoolAndUDF(), (expr,))


def bool_or(expr: Expr) -> AggregateFunction:
    """Compute logical OR across all rows.

    An aggregate function that returns `True` if the expression evaluates
    to `True` for at least one row. Returns `False` only if all rows
    evaluate to `False`. In other words, this is an existential quantifier.

    Args:
        expr: A boolean expression to evaluate for each row.

    Returns:
        An aggregate expression that computes the logical OR.

    Examples:
        Check if any review is verified:

        >>> df.aggregate([bool_or(col("verified"))])
    """
    return AggregateFunction(BoolOrUDF(), (expr,))
