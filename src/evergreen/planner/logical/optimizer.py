from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from pydantic import BaseModel

from evergreen.common.constants import (
    FIELD_NAME_REGEX,
    JSON_INDENT,
    SENTENCE_EMBEDDINGS_FIELD_SUFFIX,
)
from evergreen.common.types import FusedPlanType
from evergreen.model.embedding_model import EmbeddingType
from evergreen.planner.logical.expr import (
    AggregateFunction,
    Alias,
    BinaryExpr,
    Column,
    Expr,
    Literal,
    Not,
    Prompt,
)
from evergreen.planner.logical.plan import (
    Aggregate,
    Check,
    CountScan,
    Filter,
    FusedFilterProjection,
    LogicalPlan,
    Projection,
    RelevanceSort,
    Shuffle,
    Sort,
    TableScan,
)
from evergreen.planner.logical.types import EarlyStopComparison, Operator
from evergreen.planner.logical.udaf import (
    AggregateUDF,
    BoolAndUDF,
    BoolOrUDF,
    CountIfUDF,
    ProportionUDF,
)

if TYPE_CHECKING:
    from evergreen.core.session_state import SessionState


class ApplyOrder(Enum):
    BOTTOM_UP = auto()
    TOP_DOWN = auto()


@dataclass(frozen=True)
class Transformed[T]:
    data: T
    transformed: bool

    @classmethod
    def yes(cls, data: T) -> Transformed[T]:
        return cls(data, True)

    @classmethod
    def no(cls, data: T) -> Transformed[T]:
        return cls(data, False)


class LogicalOptimizerRule(ABC):
    @abstractmethod
    def apply_order(self) -> ApplyOrder:
        pass

    @abstractmethod
    def rewrite(
        self, plan: LogicalPlan, session_state: SessionState
    ) -> Transformed[LogicalPlan]:
        pass


SIMILARITY_FILTER_PROMPT_TEMPLATE = """<instructions>
You are an expert at information retrieval and semantic search.
You are helping optimize a database query
by finding the most relevant tuples that satisfy the predicate below.

Predicate: "{predicate}"

The predicate references a text field `{field_name}` using the syntax {{{field_name}}}.

We want to find tuples where the predicate is TRUE.

Your goal is to output a natural language query that finds such tuples
via embedding similarity.

IMPORTANT guidelines:
- Include terms likely to appear in the text field of matching tuples.
- Add synonyms and domain-specific variations
- Keep the query focused but vocabulary-rich

First, think step by step using your expertise in information retrieval.
Then, output your response.
Be specific and concise based on the predicate's semantic meaning.
</instructions>

<response_schema>
Output your response according to the following JSON schema:
{json_schema}
</response_schema>"""


class SimilarityQuery(BaseModel):
    reasoning: str
    query_text: str


SIMILARITY_QUERY_JSON_SCHEMA = SimilarityQuery.model_json_schema()
SIMILARITY_QUERY_JSON_SCHEMA_STR = json.dumps(
    SIMILARITY_QUERY_JSON_SCHEMA, indent=JSON_INDENT
)


class InsertSimilarityFilter(LogicalOptimizerRule):
    """Insert a similarity pre-filter to skip expensive LLM calls for dissimilar rows.

    For Filter or FusedFilterProjection with boolean Prompt predicates, this rule
    inserts a cheap embedding dot product check. Rows that fail the similarity
    threshold are excluded before expensive LLM evaluation.

    Uses cosine similarity (dot product of normalized embeddings) between the
    row's embedding and a query embedding derived from the prompt string.

    Applies when:
    - similarity_threshold is configured via enable_similarity_filter()
    - Plan is Filter(Prompt(..., bool)) or FusedFilterProjection with filter prompts
    - Prompt references exactly one text field (e.g., "Is {review} positive?")
    - No similarity filter exists below (traverses through Sort, Shuffle, etc.)

    For stacked unfused filters (e.g., Filter(A) -> Filter(B)), each prompt gets
    its own similarity pre-filter. The duplicate check stops at non-similarity
    Filter boundaries to allow this.

    Example (single filter):
        Filter(prompt("Is {review} positive?"))     Filter(prompt(...))
            |                                   =>      |
        TableScan                                   Filter(review_vec ⋅ query >= 0.35)
                                                        |
                                                    TableScan

    Example (stacked filters):
        Filter(prompt_A)                            Filter(prompt_A)
            |                                           |
        Filter(prompt_B)                    =>      Filter(similarity_A >= 0.35)
            |                                           |
        TableScan                                   Filter(prompt_B)
                                                        |
                                                    Filter(similarity_B >= 0.35)
                                                        |
                                                    TableScan
    """

    def apply_order(self) -> ApplyOrder:
        return ApplyOrder.BOTTOM_UP

    def rewrite(
        self, plan: LogicalPlan, session_state: SessionState
    ) -> Transformed[LogicalPlan]:
        similarity_threshold = session_state.similarity_threshold()
        if similarity_threshold is None:
            return Transformed[LogicalPlan].no(plan)

        match plan:
            case Filter(Prompt(prompt_str, return_type), input) if (
                return_type is bool and not self._has_similarity_filter_below(input)
            ):
                new_input = self._insert_similarity_filter(
                    prompt_str, similarity_threshold, input, session_state
                )

                if new_input is input:
                    return Transformed[LogicalPlan].no(plan)

                return Transformed[LogicalPlan].yes(plan.with_inputs((new_input,)))
            case FusedFilterProjection(exprs, plan_types, input) if (
                not self._has_similarity_filter_below(input)
            ):
                prompt_str = self._find_first_filter_prompt(exprs, plan_types)
                if prompt_str is None:
                    return Transformed[LogicalPlan].no(plan)

                new_input = self._insert_similarity_filter(
                    prompt_str, similarity_threshold, input, session_state
                )

                if new_input is input:
                    return Transformed[LogicalPlan].no(plan)

                return Transformed[LogicalPlan].yes(plan.with_inputs((new_input,)))
            case _:
                return Transformed[LogicalPlan].no(plan)

    @classmethod
    def _has_similarity_filter_below(cls, plan: LogicalPlan) -> bool:
        match plan:
            case Filter(BinaryExpr(BinaryExpr(_, Operator.MAX_DOT))):
                return True
            case Filter() | TableScan():
                # Hit Filter boundary or TableScan
                return False
            case _:
                inputs = plan.inputs()
                assert len(inputs) == 1
                return cls._has_similarity_filter_below(inputs[0])

    @staticmethod
    def _find_first_filter_prompt(
        exprs: tuple[Expr, ...], plan_types: tuple[FusedPlanType, ...]
    ) -> str | None:
        for expr, plan_type in zip(exprs, plan_types, strict=True):
            if plan_type == FusedPlanType.FILTER:
                match expr:
                    case Prompt(prompt_str, return_type) if return_type is bool:
                        return prompt_str
                    case _:
                        pass
        return None

    @classmethod
    def _insert_similarity_filter(
        cls,
        prompt_str: str,
        similarity_threshold: float,
        plan: LogicalPlan,
        session_state: SessionState,
    ) -> LogicalPlan:
        query_vector = cls.generate_query_vector(prompt_str, session_state)

        if query_vector is None:
            return plan

        field_names = re.findall(FIELD_NAME_REGEX, prompt_str)
        embedding_column_name = field_names[0] + SENTENCE_EMBEDDINGS_FIELD_SUFFIX

        max_dot_expr = BinaryExpr(
            Column(embedding_column_name), Operator.MAX_DOT, Literal(query_vector)
        )
        predicate = BinaryExpr(max_dot_expr, Operator.GE, Literal(similarity_threshold))

        return Filter(predicate, plan)

    @staticmethod
    def generate_query_vector(
        prompt_str: str, session_state: SessionState
    ) -> list[float] | None:
        field_names = re.findall(FIELD_NAME_REGEX, prompt_str)

        if len(field_names) != 1:
            return None

        text_column_name = field_names[0]
        prompt = SIMILARITY_FILTER_PROMPT_TEMPLATE.format(
            field_name=text_column_name,
            predicate=prompt_str,
            json_schema=SIMILARITY_QUERY_JSON_SCHEMA_STR,
        )

        similarity_query = session_state.optimizer_language_model().prompt_with_schema(
            prompt,
            SimilarityQuery,
            SIMILARITY_QUERY_JSON_SCHEMA,
        )

        query_vector = session_state.embedding_model().embed(
            similarity_query.query_text, EmbeddingType.QUERY
        )

        return query_vector


class FuseFilterProjectionPrompts(LogicalOptimizerRule):
    """Fuse consecutive Filter/Projection operators with Prompt expressions into
    a single FusedFilterProjection to reduce LLM calls.

    Only creates fusion when there are at least 2 Prompt expressions total.
    Non-Prompt expressions in Projections are preserved.

    Example:
        Projection([col_a, prompt_b])    FusedFilterProjection(
            |                         =>     exprs=[prompt_filter, col_a, prompt_b],
        Filter(prompt_filter)                plan_types=[FILTER, PROJECTION, PROJECTION]
            |                            )
        TableScan                              |
                                            TableScan
    """

    def apply_order(self) -> ApplyOrder:
        return ApplyOrder.TOP_DOWN

    def rewrite(
        self, plan: LogicalPlan, session_state: SessionState
    ) -> Transformed[LogicalPlan]:
        if not session_state.fusion_enabled():
            return Transformed[LogicalPlan].no(plan)

        exprs: list[Expr] = []
        plan_types: list[FusedPlanType] = []
        prompt_count = 0
        current_plan = plan

        # Collect consecutive Filter/Projection with Prompts
        while True:
            match current_plan:
                case Filter(predicate, input) if self._has_prompt(predicate):
                    exprs.insert(0, predicate)
                    plan_types.insert(0, FusedPlanType.FILTER)
                    prompt_count += 1
                    current_plan = input
                case Projection(projection_exprs, input) if any(
                    self._has_prompt(expr) for expr in projection_exprs
                ):
                    # Prepend all expressions from Projection (preserving order)
                    exprs = list(projection_exprs) + exprs
                    plan_types = [FusedPlanType.PROJECTION] * len(
                        projection_exprs
                    ) + plan_types

                    prompt_count += sum(
                        1 for e in projection_exprs if self._has_prompt(e)
                    )
                    current_plan = input
                case _:
                    break

        # Need at least 2 Prompts to fuse
        if prompt_count < 2:
            return Transformed[LogicalPlan].no(plan)

        return Transformed[LogicalPlan].yes(
            FusedFilterProjection(
                tuple(exprs),
                tuple(plan_types),
                current_plan,
            )
        )

    @classmethod
    def _has_prompt(cls, expr: Expr) -> bool:
        match expr:
            case Prompt():
                return True
            case Alias(inner_expr):
                return cls._has_prompt(inner_expr)
            case _:
                # TODO: We should eventually support other expressions like
                # BinaryExpr, Not, etc.
                return False


class InsertRelevanceSort(LogicalOptimizerRule):
    """Insert a RelevanceSort operator to accelerate early stopping for
    low-threshold quantifiers.

    For claims requiring few positive (or negative) examples, sorting data by
    relevance surfaces matches faster than random sampling, enabling earlier
    termination. This trades off confidence interval validity for deterministic
    speedup.

    The rule uses an LLM to generate:
    - A semantic query for embedding similarity
    - Inclusion/exclusion keywords for lexical matching

    These signals are combined via Reciprocal Rank Fusion (RRF) to produce a
    relevance-ordered stream.

    Applies when:
    - BoolOrUDF: always (existential claims need one witness)
    - CountIfUDF:
        - prompt fusion happened (estimation is the alternative, which requires
          a CountScan for count_if that processes all the tuples through the fused
          LLM operator); OR
        - count threshold <= 32 for >= / > comparisons

    Does NOT apply when:
    - BoolAndUDF (uses estimation with confidence sequences instead of relevance sort)
    - ProportionUDF (use estimation with confidence sequences instead of relevance sort)
    - Threshold is high
    - Comparison is == or != (requires full scan or sampling)
    - Multiple aggregate expressions exist
    - No prompt is found for the aggregated column

    The RelevanceSort is pushed down to above TableScan.

    Applied bottom-up to handle nested aggregations correctly.

    Example:
        Aggregate(bool_or(is_toxic))       Aggregate(bool_or(is_toxic))
            |                          =>       |
        Projection(prompt -> is_toxic)      Projection(prompt -> is_toxic)
            |                                   |
        TableScan                           RelevanceSort(query="toxic...")
                                                |
                                            TableScan
    """

    _COUNT_THRESHOLD = 32

    def apply_order(self) -> ApplyOrder:
        return ApplyOrder.BOTTOM_UP

    def rewrite(
        self, plan: LogicalPlan, session_state: SessionState
    ) -> Transformed[LogicalPlan]:
        if not session_state.relevance_sort_enabled():
            return Transformed[LogicalPlan].no(plan)

        match plan:
            case Aggregate(agg_exprs, _, early_stop_comparisons, input):
                if input.is_or_above(RelevanceSort):
                    return Transformed[LogicalPlan].no(plan)

                # Only sort if there is a single aggregate expression
                if len(agg_exprs) != 1:
                    return Transformed[LogicalPlan].no(plan)

                # Extract the aggregate function and comparison
                udaf, args = self._extract_udaf_and_args(agg_exprs[0])
                comparison = early_stop_comparisons[0]

                # Only support single-argument UDAFs for now
                if len(args) != 1:
                    return Transformed[LogicalPlan].no(plan)

                # Check if we should insert a relevance sort
                if not self._should_insert_relevance_sort(udaf, comparison, input):
                    return Transformed[LogicalPlan].no(plan)

                # Get the optimal sort order to encourage early stopping
                sort_for_positives = self._get_sort_order(udaf, comparison)

                # Get the name of the column being aggregated
                column_name = self._get_aggregated_column_name(args[0])
                if column_name is None:
                    return Transformed[LogicalPlan].no(plan)

                # Find the prompt string for the column
                prompt_str = self._find_prompt_str_for_column(column_name, plan)
                if prompt_str is None:
                    return Transformed[LogicalPlan].no(plan)

                # Get the field names from the prompt string
                field_names = re.findall(FIELD_NAME_REGEX, prompt_str)
                if len(field_names) != 1:
                    # TODO: We should support multiple field names in the prompt string
                    return Transformed[LogicalPlan].no(plan)
                text_column_name = field_names[0]

                # Find filter prompt strings that reference the same field
                filter_prompt_strs = self._find_filter_prompt_strs(
                    text_column_name, plan
                )

                # Get the predicate string
                predicate_str = repr(args[0])

                # Generate the relevance query
                query_text, query_vector, inclusion_keywords, exclusion_keywords = (
                    self._generate_relevance_query(
                        udaf,
                        comparison,
                        column_name,
                        prompt_str,
                        filter_prompt_strs,
                        predicate_str,
                        sort_for_positives,
                        session_state,
                    )
                )

                new_input = self._push_down_relevance_sort(
                    input,
                    text_column_name,
                    query_text,
                    query_vector,
                    inclusion_keywords,
                    exclusion_keywords,
                )

                return Transformed[LogicalPlan].yes(plan.with_inputs((new_input,)))
            case _:
                return Transformed[LogicalPlan].no(plan)

    @staticmethod
    def _extract_udaf_and_args(agg_expr: Expr) -> tuple[AggregateUDF, tuple[Expr, ...]]:
        match agg_expr:
            case AggregateFunction(udaf, args) | Alias(AggregateFunction(udaf, args)):
                return udaf, args
            case _:
                raise ValueError(f"Expected AggregateFunction or Alias, got {agg_expr}")

    @classmethod
    def _should_insert_relevance_sort(
        cls,
        udaf: AggregateUDF,
        comparison: EarlyStopComparison | None,
        input: LogicalPlan,
    ) -> bool:
        match udaf:
            case BoolOrUDF():
                return True
            case BoolAndUDF():
                return False
            case ProportionUDF():
                return False
            case CountIfUDF():
                if comparison is None:
                    return False
                if not isinstance(comparison.threshold, int):
                    raise ValueError(f"Unexpected type: {type(comparison.threshold)}")
                match comparison.op:
                    case Operator.GT | Operator.GE:
                        # If fusion happened, we should use relevance sort
                        # instead of estimation
                        if input.is_or_above(FusedFilterProjection):
                            return True
                        return comparison.threshold <= cls._COUNT_THRESHOLD
                    case Operator.LT | Operator.LE | Operator.EQ | Operator.NE:
                        return False
                    case _:
                        raise ValueError(f"Unexpected operator: {comparison.op}")
            case _:
                raise ValueError(f"Unexpected UDAF: {udaf}")

    @staticmethod
    def _get_sort_order(
        udaf: AggregateUDF, comparison: EarlyStopComparison | None
    ) -> bool:
        """Return True to sort for positives, False for negatives."""
        match udaf:
            case BoolOrUDF():
                # Find witnesses (positives) to confirm existence
                return True
            case CountIfUDF():
                # Find positives to prove threshold reached (only called for > and >=)
                return True
            case _:
                raise ValueError(f"Unexpected UDAF: {udaf}")

    @classmethod
    def _get_aggregated_column_name(cls, arg: Expr) -> str | None:
        match arg:
            case Column(name):
                return name
            case BinaryExpr(left, _, right):
                left_column_name = cls._get_aggregated_column_name(left)
                right_column_name = cls._get_aggregated_column_name(right)
                if left_column_name is not None and right_column_name is not None:
                    return None
                return left_column_name or right_column_name
            case Not(expr):
                return cls._get_aggregated_column_name(expr)
            case _:
                return None

    @classmethod
    def _find_prompt_str_for_column(
        cls, column_name: str, plan: LogicalPlan
    ) -> str | None:
        match plan:
            case Projection(exprs, input):
                prompt_str = cls._find_prompt_in_exprs(column_name, exprs)
                if prompt_str is not None:
                    return prompt_str
                # Continue searching in the input plan
                return cls._find_prompt_str_for_column(column_name, input)
            case FusedFilterProjection(exprs, plan_types, input):
                projection_exprs = tuple(
                    expr
                    for expr, plan_type in zip(exprs, plan_types, strict=True)
                    if plan_type == FusedPlanType.PROJECTION
                )
                prompt_str = cls._find_prompt_in_exprs(column_name, projection_exprs)
                if prompt_str is not None:
                    return prompt_str
                # Continue searching in the input plan
                return cls._find_prompt_str_for_column(column_name, input)
            case _:
                for input in plan.inputs():
                    prompt_str = cls._find_prompt_str_for_column(column_name, input)
                    if prompt_str is not None:
                        return prompt_str
        return None

    @staticmethod
    def _find_prompt_in_exprs(column_name: str, exprs: tuple[Expr, ...]) -> str | None:
        for expr in exprs:
            match expr:
                case Alias(Prompt(prompt_str), alias_name) if alias_name == column_name:
                    return prompt_str
                case _:
                    pass
        return None

    @classmethod
    def _find_filter_prompt_strs(cls, column_name: str, plan: LogicalPlan) -> list[str]:
        filter_prompt_strs: list[str] = []

        match plan:
            case Filter(predicate, input):
                match predicate:
                    case Prompt(prompt_str):
                        field_names = re.findall(FIELD_NAME_REGEX, prompt_str)
                        if column_name in field_names:
                            filter_prompt_strs.append(prompt_str)
                    case _:
                        pass
                filter_prompt_strs.extend(
                    cls._find_filter_prompt_strs(column_name, input)
                )
            case FusedFilterProjection(exprs, plan_types, input):
                for expr, plan_type in zip(exprs, plan_types, strict=True):
                    if plan_type == FusedPlanType.FILTER:
                        match expr:
                            case Prompt(prompt_str):
                                field_names = re.findall(FIELD_NAME_REGEX, prompt_str)
                                if column_name in field_names:
                                    filter_prompt_strs.append(prompt_str)
                            case _:
                                pass
                filter_prompt_strs.extend(
                    cls._find_filter_prompt_strs(column_name, input)
                )
            case _:
                for input in plan.inputs():
                    filter_prompt_strs.extend(
                        cls._find_filter_prompt_strs(column_name, input)
                    )

        return filter_prompt_strs

    @staticmethod
    def _generate_relevance_query(
        udaf: AggregateUDF,
        comparison: EarlyStopComparison | None,
        column_name: str,
        prompt_str: str,
        filter_prompt_strs: list[str],
        predicate_str: str,
        sort_for_positives: bool,
        session_state: SessionState,
    ) -> tuple[str, list[float], list[str], list[str]]:
        match udaf:
            case BoolOrUDF():
                quantifier = "there exist a tuple that satisfies"
            case CountIfUDF():
                assert comparison is not None
                quantifier = (
                    f"{comparison.op.value} {comparison.threshold} tuples satisfy"
                )
            case _:
                raise ValueError(f"Unexpected UDAF: {udaf}")

        target = "satisfy" if sort_for_positives else "violate"
        direction = "positive" if sort_for_positives else "negative"

        if filter_prompt_strs:
            filter_context = RELEVANCE_QUERY_FILTER_CONTEXT_TEMPLATE.format(
                filter_prompts="\n".join(
                    f"- {prompt_str}" for prompt_str in filter_prompt_strs
                ),
            )
        else:
            filter_context = ""

        prompt = RELEVANCE_QUERY_PROMPT_TEMPLATE.format(
            filter_context=filter_context,
            column_name=column_name,
            prompt=prompt_str,
            quantifier=quantifier,
            predicate=predicate_str,
            target=target,
            direction=direction,
            json_schema=RELEVANCE_QUERY_JSON_SCHEMA_STR,
        )

        relevance_query = session_state.optimizer_language_model().prompt_with_schema(
            prompt,
            RelevanceQuery,
            RELEVANCE_QUERY_JSON_SCHEMA,
        )

        query_vector = session_state.embedding_model().embed(
            relevance_query.query_text, EmbeddingType.QUERY
        )

        return (
            relevance_query.query_text,
            query_vector,
            relevance_query.inclusion_keywords,
            relevance_query.exclusion_keywords,
        )

    @classmethod
    def _push_down_relevance_sort(
        cls,
        plan: LogicalPlan,
        text_column_name: str,
        query_text: str,
        query_vector: list[float],
        inclusion_keywords: list[str],
        exclusion_keywords: list[str],
    ) -> LogicalPlan:
        # Reached TableScan: insert RelevanceSort above it
        if isinstance(plan, TableScan):
            return RelevanceSort(
                text_column_name,
                query_text,
                query_vector,
                inclusion_keywords,
                exclusion_keywords,
                plan,
            )

        inputs = plan.inputs()
        assert len(inputs) == 1
        new_input = cls._push_down_relevance_sort(
            inputs[0],
            text_column_name,
            query_text,
            query_vector,
            inclusion_keywords,
            exclusion_keywords,
        )
        return plan.with_inputs((new_input,))


RELEVANCE_QUERY_FILTER_CONTEXT_TEMPLATE = """
The query first filters tuples to select those that satisfy the following
prompt predicates:
{filter_prompts}
"""

RELEVANCE_QUERY_PROMPT_TEMPLATE = """<instructions>
You are an expert at information retrieval with hybrid semantic and syntactic search.
You are helping optimize a database query by sorting tuples for early stopping.
{filter_context}
The query generates the column `{column_name}` using the following prompt:
"{prompt}".

The query then aggregates over the tuples to compute:
whether {quantifier} the predicate `{predicate}`.

We want to find tuples that {target} the predicate first.

Your goal is to output:
1. query_text: A semantic search query to find such rows via embedding similarity.
2. inclusion_keywords: Keywords/phrases likely to appear in {direction} examples.
3. exclusion_keywords: Keywords/phrases unlikely to appear in {direction} examples.

IMPORTANT guidelines for keywords:
- Keywords are matched as case-insensitive substrings, so prefer shorter, atomic terms.
- Include common variations and abbreviations.
- Prefer single words or short phrases that are likely to appear verbatim.
- Avoid overly specific multi-word phrases that rarely match exactly.

First, think step by step using your expertise in information retrieval.
Then, output your response.
Be specific and concise based on the prompt and the predicate's semantic meaning.
</instructions>

<response_schema>
Output your response according to the following JSON schema:
{json_schema}
</response_schema>"""


class RelevanceQuery(BaseModel):
    reasoning: str
    query_text: str
    inclusion_keywords: list[str]
    exclusion_keywords: list[str]


RELEVANCE_QUERY_JSON_SCHEMA = RelevanceQuery.model_json_schema()
RELEVANCE_QUERY_JSON_SCHEMA_STR = json.dumps(
    RELEVANCE_QUERY_JSON_SCHEMA, indent=JSON_INDENT
)


class InsertShuffleForEstimation(LogicalOptimizerRule):
    """Insert a Shuffle operator for estimation validity in confidence sequences.

    Confidence sequences require exchangeable (randomized) data order for valid
    statistical guarantees. This rule inserts a Shuffle operator when estimation
    is enabled.

    For ungrouped aggregates:
    - If RelevanceSort is below, no shuffle is inserted, since the shuffle would
      invalidate the relevance order
    - Otherwise, global shuffle is inserted above the table scan

    For grouped aggregates:
    - Shuffle is inserted ABOVE the Sort operator
    - Uses the deepest group expressions for hierarchical shuffling
    - Groups remain contiguous after shuffling
    - If RelevanceSort is present, shuffle_rows=False preserves row order
      for base rows so that the relevance order is preserved

    Applied bottom-up to handle nested aggregations correctly.

    Example (grouped):
        Aggregate(group_by=[a, b])          Aggregate(group_by=[a, b])
            |                          =>       |
        Sort([a, b])                        Shuffle([a, b])
            |                                   |
        TableScan                           Sort([a, b])
                                                |
                                            TableScan

    Example (ungrouped):
        Aggregate(proportion(...))          Aggregate(proportion(...))
            |                          =>       |
        TableScan                           Shuffle()
                                                |
                                            TableScan
    """

    def apply_order(self) -> ApplyOrder:
        return ApplyOrder.BOTTOM_UP

    def rewrite(
        self, plan: LogicalPlan, session_state: SessionState
    ) -> Transformed[LogicalPlan]:
        if session_state.confidence_level() is None:
            return Transformed[LogicalPlan].no(plan)

        match plan:
            case Aggregate(agg_exprs, group_exprs, _, input):
                if input.is_or_above(Shuffle) or not self._supports_estimation(
                    agg_exprs, input
                ):
                    return Transformed[LogicalPlan].no(plan)

                shuffle_exprs = group_exprs or self._get_deepest_group_exprs(input)

                # Current aggregate is the only one in the plan
                if shuffle_exprs is None:
                    # If a RelevanceSort is below, we should not insert a Shuffle,
                    # since the Shuffle will invalidate the relevance order
                    if input.is_or_above(RelevanceSort):
                        return Transformed[LogicalPlan].no(plan)

                    # Otherwise, push down the Shuffle to above the TableScan
                    new_input = self._push_down_shuffle(input, session_state)
                    return Transformed[LogicalPlan].yes(plan.with_inputs((new_input,)))

                # If RelevanceSort is below, we should not shuffle the base rows,
                # since the Shuffle will invalidate the relevance order
                shuffle_rows = not input.is_or_above(RelevanceSort)

                # Insert Shuffle above the Sort for the group expressions
                new_input = self._insert_shuffle_above_sort(
                    shuffle_exprs, shuffle_rows, input, session_state
                )
                return Transformed[LogicalPlan].yes(plan.with_inputs((new_input,)))
            case _:
                return Transformed[LogicalPlan].no(plan)

    @staticmethod
    def _supports_estimation(agg_exprs: tuple[Expr, ...], input: LogicalPlan) -> bool:
        is_above_relevance_sort = input.is_or_above(RelevanceSort)
        is_above_aggregate = input.is_or_above(Aggregate)

        shuffle_is_effective = not is_above_relevance_sort or is_above_aggregate

        for expr in agg_exprs:
            match expr:
                case AggregateFunction(udaf) | Alias(AggregateFunction(udaf)):
                    if udaf.supports_estimation(has_shuffle=shuffle_is_effective):
                        return True
                case _:
                    pass
        return False

    @classmethod
    def _get_deepest_group_exprs(cls, plan: LogicalPlan) -> tuple[Expr, ...] | None:
        match plan:
            case Aggregate(_, group_exprs, _, input):
                deepest_group_exprs = cls._get_deepest_group_exprs(input)
                if deepest_group_exprs is not None:
                    return deepest_group_exprs
                return group_exprs if group_exprs else None
            case _:
                for input in plan.inputs():
                    deepest_group_exprs = cls._get_deepest_group_exprs(input)
                    if deepest_group_exprs is not None:
                        return deepest_group_exprs
                return None

    @classmethod
    def _insert_shuffle_above_sort(
        cls,
        group_exprs: tuple[Expr, ...],
        shuffle_rows: bool,
        plan: LogicalPlan,
        session_state: SessionState,
    ) -> LogicalPlan:
        match plan:
            case Sort(exprs) if [str(e) for e in exprs] == [
                str(e) for e in group_exprs
            ]:
                # Found Sort for this aggregate's group expressions, so insert
                # Shuffle above it with matching expressions
                return Shuffle(
                    group_exprs, shuffle_rows, session_state.random_seed(), plan
                )
            case _:
                inputs = plan.inputs()
                assert inputs and len(inputs) == 1
                new_input = cls._insert_shuffle_above_sort(
                    group_exprs, shuffle_rows, inputs[0], session_state
                )
                return plan.with_inputs((new_input,))

    @classmethod
    def _push_down_shuffle(
        cls, plan: LogicalPlan, session_state: SessionState
    ) -> LogicalPlan:
        # Reached TableScan: insert Shuffle above it
        if isinstance(plan, TableScan):
            return Shuffle((), True, session_state.random_seed(), plan)

        inputs = plan.inputs()
        assert len(inputs) == 1
        new_input = cls._push_down_shuffle(inputs[0], session_state)
        return plan.with_inputs((new_input,))


class InsertSortForGroupByAggregate(LogicalOptimizerRule):
    """Insert a Sort operator to ensure grouped input for StreamAggregate.

    StreamAggregate requires input rows to be sorted by the group key so that
    all rows in a group arrive consecutively. This rule inserts a Sort operator
    below any Aggregate that has GROUP BY expressions.

    The Sort is pushed down as far as possible to minimize the data volume being
    sorted. The rule stops pushing when:
    - A TableScan is reached (no further to push)
    - A Filter is reached (Sort needs to accurately count rows since it is a
      TotalCountProvider)
    - A RelevanceSort is reached (stable Sort above preserves relevance order
      within groups for early stopping)
    - The group columns don't exist in the input schema (must sort after they're
      computed)
    - An existing Sort already covers the required group expressions (as a prefix)

    Applied bottom-up to handle nested aggregations correctly.

    Example:
        Aggregate(group_by=[col_a])      Aggregate(group_by=[col_a])
            |                       =>       |
        Projection                       Projection
            |                                |
        TableScan                        Sort(col_a)
                                             |
                                         TableScan
    """

    def apply_order(self) -> ApplyOrder:
        return ApplyOrder.BOTTOM_UP

    def rewrite(
        self, plan: LogicalPlan, session_state: SessionState
    ) -> Transformed[LogicalPlan]:
        # Note: if operator fusion is enabled and we're grouping by columns computed
        # by LLM operators, Sort must be placed above the LLM Projection. Since Sort
        # is blocking, all LLM calls happen before Sort outputs, so early stopping
        # won't reduce LLM calls.
        match plan:
            case Aggregate(_, group_exprs, _, input) if (
                group_exprs and not self._has_matching_sort_below(group_exprs, input)
            ):
                field_names = {str(expr) for expr in group_exprs}

                new_input = self._push_down_sort(group_exprs, field_names, input)

                if new_input is input:
                    return Transformed[LogicalPlan].no(plan)

                return Transformed[LogicalPlan].yes(plan.with_inputs((new_input,)))
            case _:
                return Transformed[LogicalPlan].no(plan)

    @classmethod
    def _has_matching_sort_below(
        cls, group_exprs: tuple[Expr, ...], plan: LogicalPlan
    ) -> bool:
        match plan:
            case Sort(exprs):
                return cls._is_matching_sort(group_exprs, exprs)
            case RelevanceSort() | TableScan():
                # Boundaries reached that don't provide required sort order
                return False
            case _:
                inputs = plan.inputs()
                assert len(inputs) == 1
                return cls._has_matching_sort_below(group_exprs, inputs[0])

    @staticmethod
    def _is_matching_sort(
        group_exprs: tuple[Expr, ...], sort_exprs: tuple[Expr, ...]
    ) -> bool:
        """Check if the group expressions is a prefix of sort expressions."""
        group_expr_strs = [str(expr) for expr in group_exprs]
        sort_expr_strs = [str(expr) for expr in sort_exprs]

        return (
            len(group_expr_strs) <= len(sort_expr_strs)
            and sort_expr_strs[: len(group_expr_strs)] == group_expr_strs
        )

    @classmethod
    def _push_down_sort(
        cls, group_exprs: tuple[Expr, ...], field_names: set[str], plan: LogicalPlan
    ) -> LogicalPlan:
        match plan:
            case Sort(exprs) if cls._is_matching_sort(group_exprs, exprs):
                return plan
            case RelevanceSort() | Filter() | TableScan():
                # Stopping conditions: insert Sort this plan node
                return Sort(group_exprs, False, plan)
            case _:
                inputs = plan.inputs()
                assert len(inputs) == 1
                input = inputs[0]
                input_field_names = set(input.schema().field_names())

                if field_names <= input_field_names:
                    # Fields exist in the input, so push down Sort to the input
                    new_input = cls._push_down_sort(group_exprs, field_names, input)
                    if new_input is input:
                        # No change, so don't transform the plan
                        return plan
                    return plan.with_inputs((new_input,))

                # Fields don't exist in the input, so insert Sort above this plan
                return Sort(group_exprs, False, plan)


class InsertCountScanForAggregate(LogicalOptimizerRule):
    """Insert a CountScan to provide total row count for early stopping.

    For single-level aggregates (no GROUP BY), knowing the total row count helps
    with (deterministic and probabilistic) early stopping. For example (in the
    deterministic case), if checking "proportion >= 0.9" with 10 total rows:
    after seeing 2 negatives, at most 8 can be positive, so the max proportion
    is 0.8 < 0.9 and we can stop early. For the probabilistic case, knowing the
    total count allows us to sample without replacement, resulting in tighter
    confidence intervals.

    CountScan buffers all input rows and provides the count to downstream
    operators via `TotalCountProvider`. The scan is pushed down to minimize
    buffering, but stops at Filter nodes since filtering affects the count.

    This rule does NOT apply when:
    - Early stopping is disabled
    - The aggregate has GROUP BY: Sort provides counts per group instead
    - There's already a CountScan below
    - There's already an Aggregate below: nested agg has its own counting
    - Operator fusion is enabled: since counting will result in calling LLM
      operators for the entire dataset. Instead, we omit CountScan and rely on
      the early stopping conditions that do not require the total count.
    - There's a semantic filter (Filter with Prompt) below (and operator fusion
      did not happen): since CountScan would execute the filter on all tuples.
      Here, we also rely on the early stopping conditions that do not require the
      total count.

    Applied bottom-up to process inner aggregates first.

    Example:
        Aggregate(count_if(...))         Aggregate(count_if(...))
            |                     =>         |
        Projection                       Projection
            |                                |
        Filter(<structured predicate>)   CountScan
            |                                |
        TableScan                        Filter(<structured predicate>)
                                             |
                                         TableScan
    """

    def apply_order(self) -> ApplyOrder:
        return ApplyOrder.BOTTOM_UP

    def rewrite(
        self, plan: LogicalPlan, session_state: SessionState
    ) -> Transformed[LogicalPlan]:
        match plan:
            case Aggregate(_, group_exprs, _, input) if (
                session_state.early_stop_enabled()
                and not group_exprs
                and not input.is_or_above(CountScan)
                and not input.is_or_above(Aggregate)
                and not input.is_or_above(FusedFilterProjection)
                and not self._has_semantic_filter_below(input)
            ):
                new_input = self._push_down_count_scan(input)

                if new_input is input:
                    return Transformed[LogicalPlan].no(plan)

                return Transformed[LogicalPlan].yes(plan.with_inputs((new_input,)))
            case _:
                return Transformed[LogicalPlan].no(plan)

    @classmethod
    def _has_semantic_filter_below(cls, plan: LogicalPlan) -> bool:
        match plan:
            case Filter(Prompt()):
                return True
            case _:
                for input in plan.inputs():
                    if cls._has_semantic_filter_below(input):
                        return True
                return False

    @classmethod
    def _push_down_count_scan(cls, plan: LogicalPlan) -> LogicalPlan:
        if isinstance(plan, (TableScan, Filter)):
            return CountScan(plan)

        inputs = plan.inputs()
        assert len(inputs) == 1
        input = inputs[0]
        new_input = cls._push_down_count_scan(input)
        if new_input is input:
            return plan
        return plan.with_inputs((new_input,))


class PushDownEarlyStopComparison(LogicalOptimizerRule):
    """Extract and push down comparison predicates to enable early stopping.

    This rule identifies comparison predicates (e.g., `count >= 10`, `proportion
    < 0.5`) from downstream operators (e.g., Check nodes or aggregate function
    arguments), and attaches them to the corresponding Aggregate node as
    EarlyStopComparison hints.

    With these hints, accumulators can stop processing early when:
    - The result is deterministically known (e.g., threshold already exceeded)
    - A confidence interval conclusively satisfies/violates the comparison

    Extraction:
        The rule extracts comparisons of the form `Column op Literal` or
        `Literal op Column`. For compound predicates with AND/OR, comparisons
        are extracted recursively from both sides. Each extracted comparison
        is matched to its corresponding aggregate expression by column name.

    Limitations:
        The aggregate's early stopping logic uses AND semantics: all
        accumulators must be able to stop before the aggregate stops. This
        works correctly for AND predicates but does not implement true OR
        semantics (stop when any condition is definitively satisfied).

        However, extracting comparisons from OR predicates is still beneficial:
        1. It enables early stopping with AND semantics, which is still better
           than no early stopping (without comparisons, accumulators cannot
           stop early at all).
        2. It can trigger confidence interval computation in accumulators, if
        estimation is enabled.

    Applied top-down so outer comparisons are pushed before processing children.

    Example (single comparison):
        Check(cnt >= 100)           Check(cnt >= 100)
            |                =>                |
        Aggregate(                         Aggregate(
            agg_exprs=[count_if(...)],         agg_exprs=[count_if(...)],
            early_stop_comparisons=[None]      early_stop_comparisons=[(cnt, >=, 100)]
        )                                  )

    Example (compound comparison):
        Check((prop > 0.5) AND (cnt > 30))
            |
        Aggregate(
            agg_exprs=[proportion(...), count_if(...)],
            early_stop_comparisons=[None, None]
        )
                        |
                        V
        Check((prop > 0.5) AND (cnt > 30))
            |
        Aggregate(
            agg_exprs=[proportion(...), count_if(...)],
            early_stop_comparisons=[(prop, >, 0.5), (cnt, >, 30)]
        )
    """

    def apply_order(self) -> ApplyOrder:
        return ApplyOrder.TOP_DOWN

    def rewrite(
        self, plan: LogicalPlan, session_state: SessionState
    ) -> Transformed[LogicalPlan]:
        comparisons = self._extract_comparison_from_plan(plan)
        if not comparisons:
            return Transformed[LogicalPlan].no(plan)

        if isinstance(plan, TableScan):
            return Transformed[LogicalPlan].no(plan)

        inputs = plan.inputs()
        assert len(inputs) == 1
        new_input = inputs[0]
        for comparison in comparisons:
            new_input = self._push_down_to_aggregate(comparison, new_input)

        if new_input is inputs[0]:
            return Transformed[LogicalPlan].no(plan)

        return Transformed[LogicalPlan].yes(plan.with_inputs((new_input,)))

    @classmethod
    def _extract_comparison_from_plan(
        cls, plan: LogicalPlan
    ) -> list[EarlyStopComparison]:
        match plan:
            case Check(predicate):
                return cls._extract_comparison(predicate)
            case Aggregate(agg_exprs):
                comparisons: list[EarlyStopComparison] = []
                for agg_expr in agg_exprs:
                    match agg_expr:
                        case AggregateFunction(_, args) | Alias(
                            AggregateFunction(_, args)
                        ):
                            for arg in args:
                                comparisons.extend(cls._extract_comparison(arg))
                        case _:
                            pass
                return comparisons
            # TODO: Handle other downstream operators like Projection.
            case _:
                pass

        return []

    @classmethod
    def _extract_comparison(cls, predicate: Expr) -> list[EarlyStopComparison]:
        match predicate:
            case BinaryExpr(Column(name), op, Literal(value)):
                return [EarlyStopComparison(name, op, value)]
            case BinaryExpr(Literal(value), op, Column(name)):
                return [EarlyStopComparison(name, op.flip(), value)]
            case BinaryExpr(left, Operator.AND | Operator.OR, right):
                return cls._extract_comparison(left) + cls._extract_comparison(right)
            case _:
                return []

    @classmethod
    def _push_down_to_aggregate(
        cls, comparison: EarlyStopComparison, plan: LogicalPlan
    ) -> LogicalPlan:
        match plan:
            case Aggregate(agg_exprs, group_exprs, early_stop_comparisons, input):
                new_early_stop_comparisons: list[EarlyStopComparison | None] = []
                transformed = False

                for agg_expr, early_stop_comparison in zip(
                    agg_exprs, early_stop_comparisons, strict=True
                ):
                    column_name = agg_expr.to_field(input.schema()).name
                    if (
                        comparison.column_name == column_name
                        and early_stop_comparison is None
                    ):
                        new_early_stop_comparisons.append(comparison)
                        transformed = True
                    else:
                        new_early_stop_comparisons.append(early_stop_comparison)

                if transformed:
                    return Aggregate(
                        agg_exprs, group_exprs, tuple(new_early_stop_comparisons), input
                    )
            case _:
                pass

        return plan


class LogicalOptimizer:
    _MAX_ITERATIONS = 10

    def __init__(self) -> None:
        self._rules = [
            # Must be before InsertRelevanceSort, which needs early stop comparisons
            PushDownEarlyStopComparison(),
            # Must be before InsertRelevanceSort, InsertCountScanForAggregate,
            # and InsertSimilarityFilter
            FuseFilterProjectionPrompts(),
            # Must be after PushDownEarlyStopComparison and FuseFilterProjectionPrompts
            # Must be before InsertShuffleForEstimation
            InsertRelevanceSort(),
            # Must be before InsertShuffleForEstimation
            InsertSortForGroupByAggregate(),
            # Must be after InsertRelevanceSort and InsertSortForGroupByAggregate
            InsertShuffleForEstimation(),
            # Must be after FuseFilterProjectionPrompts
            InsertCountScanForAggregate(),
            # Must be after FuseFilterProjectionPrompts
            InsertSimilarityFilter(),
        ]

    def optimize(self, plan: LogicalPlan, session_state: SessionState) -> LogicalPlan:
        for _ in range(self._MAX_ITERATIONS):
            transformed = False

            for rule in self._rules:
                result = self._apply_rule(rule, plan, session_state)

                if result.transformed:
                    plan = result.data
                    transformed = True

            if not transformed:
                break

        return plan

    def _apply_rule(
        self, rule: LogicalOptimizerRule, plan: LogicalPlan, session_state: SessionState
    ) -> Transformed[LogicalPlan]:
        match rule.apply_order():
            case ApplyOrder.BOTTOM_UP:
                return self._apply_bottom_up(rule, plan, session_state)
            case ApplyOrder.TOP_DOWN:
                return self._apply_top_down(rule, plan, session_state)

    def _apply_bottom_up(
        self, rule: LogicalOptimizerRule, plan: LogicalPlan, session_state: SessionState
    ) -> Transformed[LogicalPlan]:
        transformed = False
        new_inputs: list[LogicalPlan] = []

        for input in plan.inputs():
            result = self._apply_bottom_up(rule, input, session_state)
            new_inputs.append(result.data)
            transformed |= result.transformed

        if transformed:
            plan = plan.with_inputs(tuple(new_inputs))

        result = rule.rewrite(plan, session_state)
        transformed |= result.transformed

        return Transformed[LogicalPlan](result.data, transformed)

    def _apply_top_down(
        self, rule: LogicalOptimizerRule, plan: LogicalPlan, session_state: SessionState
    ) -> Transformed[LogicalPlan]:
        result = rule.rewrite(plan, session_state)
        plan = result.data
        transformed = result.transformed

        new_inputs: list[LogicalPlan] = []
        for input in plan.inputs():
            result = self._apply_top_down(rule, input, session_state)
            new_inputs.append(result.data)
            transformed |= result.transformed

        if transformed:
            plan = plan.with_inputs(tuple(new_inputs))

        return Transformed[LogicalPlan](plan, transformed)
