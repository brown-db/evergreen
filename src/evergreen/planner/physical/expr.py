from __future__ import annotations

import json
import logging
import reprlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from itertools import combinations
from math import ceil, floor
from typing import TypeGuard

import evergreen.planner.logical.expr as logi_expr
from evergreen.common.constants import JSON_INDENT
from evergreen.common.types import (
    CountInterval,
    Interval,
    ProportionInterval,
)
from evergreen.model.language_model import LanguageModel, create_output_schema
from evergreen.planner.logical.expr import Operator
from evergreen.planner.logical.udaf import (
    AggregateUDF,
)
from evergreen.planner.physical.accumulator import Accumulator
from evergreen.provenance import Prov
from evergreen.storage.row import AnnotatedValue, ProvCollection, Row
from evergreen.storage.row_id import RowId

logger = logging.getLogger(__name__)


class PhysicalExpr(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def to_logical(self) -> logi_expr.Expr:
        pass

    @abstractmethod
    def evaluate(self, row: Row, row_id: RowId) -> AnnotatedValue:
        pass

    def evaluate_batch(self, rows: list[tuple[Row, RowId]]) -> list[AnnotatedValue]:
        return [self.evaluate(row, row_id) for row, row_id in rows]


class FusedPhysicalExprs(ABC):
    @abstractmethod
    def evaluate(self, row: Row, row_id: RowId) -> tuple[AnnotatedValue, ...]:
        pass

    def evaluate_batch(
        self, rows: list[tuple[Row, RowId]]
    ) -> list[tuple[AnnotatedValue, ...]]:
        return [self.evaluate(row, row_id) for row, row_id in rows]


class Column(PhysicalExpr):
    def __init__(self, name: str, index: int) -> None:
        self._name = name
        self._index = index

    def __str__(self) -> str:
        return self._name

    def to_logical(self) -> logi_expr.Expr:
        return logi_expr.Column(self._name)

    def evaluate(self, row: Row, row_id: RowId) -> AnnotatedValue:
        value = row[self._index]
        prov = row.get_prov(self._index)

        if prov is None and isinstance(value, bool):
            if value:
                return AnnotatedValue(value, Prov.pos_token(row_id, self.to_logical()))
            else:
                return AnnotatedValue(value, Prov.neg_token(row_id, self.to_logical()))

        return AnnotatedValue(value, prov)


class Literal(PhysicalExpr):
    def __init__(self, value: object) -> None:
        self._value = value

    def __str__(self) -> str:
        return reprlib.repr(self._value)

    def to_logical(self) -> logi_expr.Expr:
        return logi_expr.Literal(self._value)

    def evaluate(self, row: Row, row_id: RowId) -> AnnotatedValue:
        return AnnotatedValue(self._value, None)


class BinaryExpr(PhysicalExpr):
    def __init__(
        self,
        left: PhysicalExpr,
        op: Operator,
        right: PhysicalExpr,
        minimal_provenance: bool,
    ) -> None:
        self._left = left
        self._op = op
        self._right = right
        self._minimal_provenance = minimal_provenance

    def __str__(self) -> str:
        return f"{self._left} {self._op.value} {self._right}"

    def to_logical(self) -> logi_expr.BinaryExpr:
        return logi_expr.BinaryExpr(
            self._left.to_logical(),
            self._op,
            self._right.to_logical(),
        )

    def evaluate(self, row: Row, row_id: RowId) -> AnnotatedValue:
        left_annotated_value = self._left.evaluate(row, row_id)
        right_annotated_value = self._right.evaluate(row, row_id)

        match self._op:
            case (
                Operator.GE
                | Operator.GT
                | Operator.LT
                | Operator.LE
                | Operator.EQ
                | Operator.NE
            ):
                return self.evaluate_comparison(
                    left_annotated_value,
                    self._op,
                    right_annotated_value,
                    row_id,
                    self.to_logical(),
                    self._minimal_provenance,
                )
            case Operator.AND:
                return self._evaluate_and(
                    left_annotated_value,
                    right_annotated_value,
                    self._minimal_provenance,
                )
            case Operator.OR:
                return self._evaluate_or(
                    left_annotated_value,
                    right_annotated_value,
                    self._minimal_provenance,
                )
            case Operator.MAX_DOT:
                result = self._op.evaluate(
                    left_annotated_value.value, right_annotated_value.value
                )
                logger.debug("Row %s %s product: %s", row_id, self._op.value, result)
                return AnnotatedValue(result, None)

    @classmethod
    def evaluate_comparison(
        cls,
        left_annotated_value: AnnotatedValue,
        op: Operator,
        right_annotated_value: AnnotatedValue,
        row_id: RowId | None,
        predicate: logi_expr.Expr | None,
        minimal_provenance: bool,
    ) -> AnnotatedValue:
        left_value, left_prov = left_annotated_value.value, left_annotated_value.prov
        right_value, right_prov = (
            right_annotated_value.value,
            right_annotated_value.prov,
        )

        flipped_op = op.flip()

        values = (left_value, right_value)
        match (left_prov, right_prov):
            case (ProvCollection(), ProvCollection()):
                if cls._are_count_values(values):
                    left_value, right_value = values
                    left_result = cls._evaluate_count_comparison(
                        left_value, left_prov, right_value, op, minimal_provenance
                    )
                    right_result = cls._evaluate_count_comparison(
                        right_value,
                        right_prov,
                        left_value,
                        flipped_op,
                        minimal_provenance,
                    )
                elif cls._are_proportion_values(values):
                    left_value, right_value = values
                    left_result = cls._evaluate_proportion_comparison(
                        left_value, left_prov, right_value, op, minimal_provenance
                    )
                    right_result = cls._evaluate_proportion_comparison(
                        right_value,
                        right_prov,
                        left_value,
                        flipped_op,
                        minimal_provenance,
                    )
                else:
                    raise ValueError(
                        f"Unexpected types: {type(left_value)}, {type(right_value)}"
                    )
                return cls._evaluate_and(left_result, right_result, minimal_provenance)
            case (ProvCollection(), None):
                if cls._are_count_values(values):
                    left_value, right_value = values
                    return cls._evaluate_count_comparison(
                        left_value, left_prov, right_value, op, minimal_provenance
                    )
                elif cls._are_proportion_values(values):
                    left_value, right_value = values
                    return cls._evaluate_proportion_comparison(
                        left_value, left_prov, right_value, op, minimal_provenance
                    )
                raise ValueError(
                    f"Unexpected types: {type(left_value)}, {type(right_value)}"
                )
            case (None, ProvCollection()):
                if cls._are_count_values(values):
                    left_value, right_value = values
                    return cls._evaluate_count_comparison(
                        right_value,
                        right_prov,
                        left_value,
                        flipped_op,
                        minimal_provenance,
                    )
                elif cls._are_proportion_values(values):
                    left_value, right_value = values
                    return cls._evaluate_proportion_comparison(
                        right_value,
                        right_prov,
                        left_value,
                        flipped_op,
                        minimal_provenance,
                    )
                raise ValueError(
                    f"Unexpected types: {type(left_value)}, {type(right_value)}"
                )
            case (Prov(), None):
                value = op.evaluate(left_value, right_value)
                return AnnotatedValue(value, left_prov)
            case (None, Prov()):
                value = op.evaluate(left_value, right_value)
                return AnnotatedValue(value, right_prov)
            case (None, None):
                value = op.evaluate(left_value, right_value)
                if row_id is not None and predicate is not None:
                    if value:
                        prov = Prov.pos_token(row_id, predicate)
                    else:
                        prov = Prov.neg_token(row_id, predicate)
                    return AnnotatedValue(value, prov)
                return AnnotatedValue(value, Prov.one() if value else Prov.zero())
            case _:
                raise ValueError(
                    f"Unexpected types: {type(left_prov)}, {type(right_prov)}"
                )

    @staticmethod
    def _are_count_values(
        values: tuple[object, object],
    ) -> TypeGuard[tuple[int | CountInterval, int | CountInterval]]:
        left, right = values
        return isinstance(left, (int, CountInterval)) and isinstance(
            right, (int, CountInterval)
        )

    @staticmethod
    def _are_proportion_values(
        values: tuple[object, object],
    ) -> TypeGuard[tuple[float | ProportionInterval, float | ProportionInterval]]:
        left, right = values
        return isinstance(left, (float, ProportionInterval)) and isinstance(
            right, (float, ProportionInterval)
        )

    @classmethod
    def _evaluate_count_comparison(
        cls,
        value: float | Interval,
        prov_collection: ProvCollection,
        threshold: float | Interval,
        op: Operator,
        minimal_provenance: bool,
    ) -> AnnotatedValue:
        result = op.evaluate(value, threshold)

        if result:
            prov = cls._compute_provenance_for_counts(
                value, prov_collection, threshold, op, minimal_provenance
            )
        else:
            prov = cls._compute_provenance_for_counts(
                value, prov_collection, threshold, op.negate(), minimal_provenance
            )

        return AnnotatedValue(result, prov)

    @staticmethod
    def _compute_provenance_for_counts(
        value: float | Interval,
        prov_collection: ProvCollection,
        threshold: float | Interval,
        op: Operator,
        minimal_provenance: bool,
    ) -> Prov:
        # TODO: Figure out when to use standard polynomial vs. dual polynomial.
        # For now, we use the standard polynomial.
        # Eventually, we can compare the expected sizes.
        match op:
            case Operator.GE | Operator.GT:
                # GE / GT: Only need to consider combinations of the positive
                # polynomials, since negative polynomials are false (equal to 0)
                # and turn the inner product to 0
                if isinstance(value, Interval) or isinstance(threshold, Interval):
                    # If either value or threshold is a confidence interval,
                    # the resulting polynomial is just a single combination
                    # of the positive polynomials; hence, we set the
                    # min_positives to the number of sampled positive
                    # polynomials
                    min_positives = len(prov_collection.pos_polynomials)
                else:
                    # Minimum positive count to satisfy comparison:
                    # GE: count >= threshold requires ceil(threshold) items
                    # GT: count > threshold requires floor(threshold) + 1 items
                    min_positives = (
                        ceil(threshold) if op == Operator.GE else floor(threshold) + 1
                    )
                if minimal_provenance:
                    # Return just one witness: first min_positives items
                    combs = [prov_collection.pos_polynomials[:min_positives]]
                else:
                    # Return all combinations of the positive polynomials
                    combs = combinations(prov_collection.pos_polynomials, min_positives)
                polynomial = Prov.zero()
                for comb in combs:
                    product = Prov.one()
                    for prov in comb:
                        product *= prov
                    polynomial += product
                return polynomial
            case Operator.LE | Operator.LT | Operator.EQ | Operator.NE:
                # EQ: Only compute a single combination,
                # since the others will turn the inner product to 0

                # LE / LT: Only compute the provenance for equality, since other
                # combinations where threshold <= value / threshold < value will
                # turn the inner product to 0

                # NE: Only compute the provenance for equality,
                # since other combinations where value != threshold
                # will turn the inner product to 0
                polynomial = Prov.one()
                for prov in prov_collection.pos_polynomials:
                    polynomial *= prov
                for prov in prov_collection.neg_polynomials:
                    polynomial *= prov
                return polynomial
            case _:
                raise ValueError(f"Unexpected operator: {op}")

    @classmethod
    def _evaluate_proportion_comparison(
        cls,
        value: float | ProportionInterval,
        prov_collection: ProvCollection,
        threshold: float | ProportionInterval,
        op: Operator,
        minimal_provenance: bool,
    ) -> AnnotatedValue:
        total = prov_collection.total()

        if total == 0:
            return AnnotatedValue(op.evaluate(value, threshold), Prov.one())

        value_scaled = value * total
        threshold_scaled = threshold * total

        return cls._evaluate_count_comparison(
            value_scaled, prov_collection, threshold_scaled, op, minimal_provenance
        )

    @classmethod
    def _evaluate_and(
        cls,
        left_annotated_value: AnnotatedValue,
        right_annotated_value: AnnotatedValue,
        minimal_provenance: bool,
    ) -> AnnotatedValue:
        left_value, left_prov = left_annotated_value.value, left_annotated_value.prov
        right_value, right_prov = (
            right_annotated_value.value,
            right_annotated_value.prov,
        )

        assert not isinstance(left_prov, ProvCollection) and not isinstance(
            right_prov, ProvCollection
        )

        if not isinstance(left_value, bool) or not isinstance(right_value, bool):
            raise ValueError(
                f"Unexpected types: {type(left_value)}, {type(right_value)}"
            )

        value = left_value and right_value

        # Compute provenance
        if value:
            # Both left and right values are true:
            # combine true provenances jointly (*)
            prov = cls._merge_prov_pair(left_prov, right_prov, lambda p, q: p * q)
        elif not left_value and not right_value:
            # Both left and right values are false:
            # combine false provenances alternatively (+)
            if minimal_provenance:
                prov = left_prov if left_prov is not None else right_prov
            else:
                prov = cls._merge_prov_pair(left_prov, right_prov, lambda p, q: p + q)
        elif not left_value:
            # Only left value is false:
            # cite false left provenance
            prov = left_prov
        else:
            # Only right value is false:
            # cite false right provenance
            prov = right_prov

        return AnnotatedValue(value, prov)

    @classmethod
    def _evaluate_or(
        cls,
        left_annotated_value: AnnotatedValue,
        right_annotated_value: AnnotatedValue,
        minimal_provenance: bool,
    ) -> AnnotatedValue:
        left_value, left_prov = left_annotated_value.value, left_annotated_value.prov
        right_value, right_prov = (
            right_annotated_value.value,
            right_annotated_value.prov,
        )

        assert not isinstance(left_prov, ProvCollection) and not isinstance(
            right_prov, ProvCollection
        )

        if not isinstance(left_value, bool) or not isinstance(right_value, bool):
            raise ValueError(
                f"Unexpected types: {type(left_value)}, {type(right_value)}"
            )

        value = left_value or right_value

        # Compute provenance
        if not value:
            # Both left and right values are false:
            # combine false provenances jointly (*)
            prov = cls._merge_prov_pair(left_prov, right_prov, lambda p, q: p * q)
        elif left_value and right_value:
            # Both left and right values are true:
            # combine true provenances alternatively (+)
            if minimal_provenance:
                prov = left_prov if left_prov is not None else right_prov
            else:
                prov = cls._merge_prov_pair(left_prov, right_prov, lambda p, q: p + q)
        elif left_value:
            # Only left value is true:
            # cite true left provenance
            prov = left_prov
        else:
            # Only right value is true:
            # cite true right provenance
            prov = right_prov

        return AnnotatedValue(value, prov)

    @staticmethod
    def _merge_prov_pair(
        left_prov: Prov | None,
        right_prov: Prov | None,
        op: Callable[[Prov, Prov], Prov],
    ) -> Prov | None:
        if left_prov is not None and right_prov is not None:
            return op(left_prov, right_prov)
        elif left_prov is not None:
            return left_prov
        elif right_prov is not None:
            return right_prov
        else:
            return None


class Not(PhysicalExpr):
    def __init__(self, expr: PhysicalExpr) -> None:
        self._expr = expr

    def __str__(self) -> str:
        return f"~{self._expr}"

    def to_logical(self) -> logi_expr.Expr:
        return logi_expr.Not(self._expr.to_logical())

    def evaluate(self, row: Row, row_id: RowId) -> AnnotatedValue:
        annotated_value = self._expr.evaluate(row, row_id)

        if not isinstance(annotated_value.value, bool):
            raise ValueError(f"Unexpected type: {type(annotated_value.value)}")

        match annotated_value.prov:
            case None:
                return AnnotatedValue(not annotated_value.value, None)
            case Prov():
                value = not annotated_value.value
                if value:
                    prov = Prov.neg_token(row_id, self._expr.to_logical())
                else:
                    prov = Prov.pos_token(row_id, self._expr.to_logical())
                return AnnotatedValue(value, prov)
            case ProvCollection():
                raise ValueError(
                    f"Unexpected provenance type: {type(annotated_value.prov)}"
                )


PROMPT_TEMPLATE = """<instructions>
{instructions}
</instructions>

<field_values>
{field_values}
</field_values>

<response_schema>
Output your response according to the following JSON schema:
{json_schema}
</response_schema>"""


class Prompt(PhysicalExpr):
    def __init__(
        self,
        prompt_str: str,
        return_type: type[object],
        field_indices: tuple[tuple[str, int, str | None], ...],
        model: LanguageModel,
    ) -> None:
        self._prompt_str = prompt_str
        self._return_type = return_type
        self._field_indices = field_indices
        self._model = model

    def __str__(self) -> str:
        return f"prompt({self._prompt_str})"

    def to_logical(self) -> logi_expr.Expr:
        return logi_expr.Prompt(self._prompt_str, self._return_type)

    def prompt_str(self) -> str:
        return self._prompt_str

    def return_type(self) -> type[object]:
        return self._return_type

    def field_indices(self) -> tuple[tuple[str, int, str | None], ...]:
        return self._field_indices

    def model(self) -> LanguageModel:
        return self._model

    def evaluate(self, row: Row, row_id: RowId) -> AnnotatedValue:
        return self.evaluate_batch([(row, row_id)])[0]

    def evaluate_batch(self, rows: list[tuple[Row, RowId]]) -> list[AnnotatedValue]:
        instruction_str = self.create_instruction_str(
            self._prompt_str, self._return_type
        )

        schema = create_output_schema(self._return_type)
        json_schema = schema.model_json_schema()
        json_schema_str = json.dumps(json_schema, indent=JSON_INDENT)

        formatted_prompts = [
            PROMPT_TEMPLATE.format(
                instructions=instruction_str,
                field_values=self.create_field_values_str(row, self._field_indices),
                json_schema=json_schema_str,
            )
            for row, _ in rows
        ]

        responses = self._model.prompt(
            formatted_prompts, self._return_type, schema, json_schema
        )

        return [AnnotatedValue(response, None) for response in responses]

    @staticmethod
    def create_instruction_str(prompt_str: str, return_type: type[object]) -> str:
        instruction = prompt_str
        if return_type is bool:
            instruction = (
                f"Evaluate the predicate and output true or false: {instruction}"
            )
        return instruction

    @staticmethod
    def create_field_values_str(
        row: Row, field_indices: tuple[tuple[str, int, str | None], ...]
    ) -> str:
        blocks: list[str] = []
        for field_name, index, description in field_indices:
            open_tag = (
                f'<{field_name} description="{description}">'
                if description
                else f"<{field_name}>"
            )
            blocks.append(f"{open_tag}\n{row[index]}\n</{field_name}>")
        return "\n".join(blocks)


FUSED_INSTRUCTIONS = """
You are given a sequence of instructions to evaluate against the field values below.

Each instruction should be evaluated ONLY based on what is explicitly stated in the 
field values; do not let your answer to one instruction influence another.

The instructions form a logical sequence: if an earlier predicate evaluates to false,
later instructions in that sequence may not be meaningful, but you must still provide
a valid output for each. In such cases, output `false` for boolean types, an
empty string for text types, 0 for numeric types, and the first value for enum types.

Output the i-th result corresponding to the i-th instruction.

Example 1:
<instruction_0>
Does the {review} mention fighting scenes?
</instruction_0>
<instruction_1>
Does the {review} say fighting scenes are the best?
</instruction_1>
<field_values>
<review>
This movie is amazing! They have the best fighting scenes I've ever seen.
</review>
</field_values>
Output: {"output_0": true, "output_1": true}

Example 2:
<instruction_0>
Does the {review} mention fighting scenes?
</instruction_0>
<instruction_1>
Does the {review} say fighting scenes are the best?
</instruction_1>
<field_values>
<review>
This movie is was pretty good. I enjoyed their soundtrack.
</review>
</field_values>
Output: {"output_0": false, "output_1": false}
"""


class FusedPrompt(FusedPhysicalExprs):
    def __init__(
        self,
        prompt_strs: tuple[str, ...],
        return_types: tuple[type[object], ...],
        field_indices: tuple[tuple[str, int, str | None], ...],
        model: LanguageModel,
    ):
        self._prompt_strs = prompt_strs
        self._return_types = return_types
        self._field_indices = field_indices
        self._model = model

    def __str__(self) -> str:
        return f"fused_prompt({', '.join(self._prompt_strs)})"

    def evaluate(self, row: Row, row_id: RowId) -> tuple[AnnotatedValue, ...]:
        return self.evaluate_batch([(row, row_id)])[0]

    def evaluate_batch(
        self, rows: list[tuple[Row, RowId]]
    ) -> list[tuple[AnnotatedValue, ...]]:
        instructions: list[str] = [FUSED_INSTRUCTIONS]
        for i, (prompt_str, return_type) in enumerate(
            zip(self._prompt_strs, self._return_types, strict=True)
        ):
            instruction_str = Prompt.create_instruction_str(prompt_str, return_type)
            instructions.append(
                f"<instruction_{i}>\n{instruction_str}\n</instruction_{i}>"
            )
        instructions_str = "\n".join(instructions)

        schema = create_output_schema(self._return_types)
        json_schema = schema.model_json_schema()
        json_schema_str = json.dumps(json_schema, indent=JSON_INDENT)

        formatted_prompts = [
            PROMPT_TEMPLATE.format(
                instructions=instructions_str,
                field_values=Prompt.create_field_values_str(row, self._field_indices),
                json_schema=json_schema_str,
            )
            for row, _ in rows
        ]

        responses = self._model.fused_prompt(
            formatted_prompts, self._return_types, schema, json_schema
        )

        return [
            tuple(AnnotatedValue(value, None) for value in response)
            for response in responses
        ]


class AggregateFunctionExpr:
    def __init__(
        self,
        udaf: AggregateUDF,
        args: tuple[PhysicalExpr, ...],
        relative_error: float,
        minimal_provenance: bool,
    ) -> None:
        self._udaf = udaf
        self._args = args
        self._relative_error = relative_error
        self._minimal_provenance = minimal_provenance

        self.has_shuffle: bool = False

    def __str__(self) -> str:
        return f"{self._udaf.name()}({', '.join(str(arg) for arg in self._args)})"

    def supports_estimation(self) -> bool:
        return self._udaf.supports_estimation(self.has_shuffle)

    def create_accumulator(
        self, precomputed_total: int | None, confidence_level: float | None
    ) -> Accumulator:
        return self._udaf.create_accumulator(
            precomputed_total,
            confidence_level,
            self._relative_error,
            self._minimal_provenance,
        )

    def evaluate_args(self, row: Row, row_id: RowId) -> tuple[AnnotatedValue, ...]:
        return tuple(arg.evaluate(row, row_id) for arg in self._args)
