import json
import logging
import time

import tiktoken
from pydantic import BaseModel

from evergreen.catalog.schema import Schema
from evergreen.common.constants import JSON_INDENT
from evergreen.data_frame import QueryMetrics
from evergreen.model.config import CortexLanguageModel, CortexModelConfig
from experiments.common import CONNECTION_NAME, MODEL_CONTEXT_WINDOW_TOKENS
from experiments.types import EvaluationResult

logger = logging.getLogger(__name__)

# Claude uses a similar tokenizer to cl100k_base
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

# Safety margin as percentage of context window due to token approximation
_SAFETY_MARGIN_PERCENT = 0.10

# Cortex's max output tokens
_MAX_OUTPUT_TOKENS = CortexLanguageModel.MAX_OUTPUT_TOKENS


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def evaluate_claim(
    claim: str,
    hints: str,
    agg_prompt: str,
    dataset_path: str,
    schema: Schema,
    language_model: str,
) -> EvaluationResult:
    context_window_tokens = MODEL_CONTEXT_WINDOW_TOKENS[language_model]
    remaining_tokens = _MAX_OUTPUT_TOKENS
    used_tokens = context_window_tokens - remaining_tokens

    prompt_template_tokens = _count_tokens(
        _PROMPT_TEMPLATE.format(
            dataset="",
            claim=claim,
            agg_prompt=agg_prompt,
            schema=schema,
            hints=hints,
            json_schema=_GROUNDING_QUERY_JSON_SCHEMA_STR,
            used=used_tokens,
            total=context_window_tokens,
            remaining=remaining_tokens,
        )
    )

    usable_context = int(context_window_tokens * (1 - _SAFETY_MARGIN_PERCENT))
    max_dataset_tokens = usable_context - prompt_template_tokens - _MAX_OUTPUT_TOKENS

    model_config = CortexModelConfig(
        language_models=[language_model],
        embedding_model="",
        connection_name=CONNECTION_NAME,
    )
    model = model_config.create_language_model(cache_dir=None)

    t0 = time.perf_counter()

    row_strs: list[str] = []
    token_count = 0

    with open(dataset_path) as f:
        for line in f:
            obj = json.loads(line)
            filtered_obj = {k: v for k, v in obj.items() if k in schema.field_names()}

            row_str = json.dumps(filtered_obj)
            # "\n".join() adds N-1 separators for N rows
            row_with_sep = row_str if not row_strs else "\n" + row_str
            row_tokens = _count_tokens(row_with_sep)

            if token_count + row_tokens > max_dataset_tokens:
                logger.debug(f"Dataset truncated to {len(row_strs)} rows")
                break

            row_strs.append(row_str)
            token_count += row_tokens

    dataset_str = "\n".join(row_strs)

    prompt_str = _PROMPT_TEMPLATE.format(
        dataset=dataset_str,
        claim=claim,
        agg_prompt=agg_prompt,
        schema=schema,
        hints=hints,
        json_schema=_GROUNDING_QUERY_JSON_SCHEMA_STR,
        used=used_tokens,
        total=context_window_tokens,
        remaining=remaining_tokens,
    )
    result = model.prompt_with_schema(
        prompt_str, GroundingQuery, _GROUNDING_QUERY_JSON_SCHEMA
    )

    execution_latency = time.perf_counter() - t0

    return EvaluationResult(
        verification_result=result.grounded,
        query_metrics=QueryMetrics(
            planning_latency=0.0,
            execution_latency=execution_latency,
            language_model_metrics=model.metrics(),
            embedding_model_metrics=(),
            optimizer_language_model_metrics=(),
        ),
        filter_metrics=None,
        map_metrics=None,
        prov_tokens={},
        reasoning=result.reasoning,
    )


_PROMPT_TEMPLATE = """<instructions>
You are a fact checking expert.
You are given a <dataset> of rows and a <claim> about the dataset.
The <claim> was generated in response to the <agg_prompt> over the dataset.
The provided <schema> describes the fields of the underlying data.
Your goal is to determine if the <claim> is grounded in the <dataset>,
i.e., fully supported by evidence in the dataset.
A <claim> is NOT grounded if the <dataset> contradicts it or lacks sufficient evidence.

Steps:
1. Formally express the logical structure of the <claim>.
  - Think step by step and output your reasoning.
  - Leverage your knowledge of first-order logic and its extensions here.
  - Identify the key logical components of the <claim>, such as constants,
  variables, predicates, functions, quantifiers, etc.
  - If provided, use the <hints> below for clarification.
2. Output your final decision.
  - `true` if the claim is grounded in the dataset
  - `false` otherwise
</instructions>

<dataset>
{dataset}
</dataset>

<agg_prompt>
{agg_prompt}
</agg_prompt>

<schema>
{schema}
</schema>

<claim>
{claim}
</claim>

<hints>
{hints}
</hints>

<response_schema>
Output your response according to the following JSON schema:
{json_schema}
</response_schema>

<system_warning>Token usage: {used}/{total}; {remaining} remaining</system_warning>"""

# The system warning above is adapted from the Claude documentation below:
# https://platform.claude.com/docs/en/build-with-claude/context-windows#context-awareness-in-claude-sonnet-4-6-sonnet-4-5-and-haiku-4-5
# The token usage values reflect the context window capacity, not actual usage,
# to signal that the model is operating near its context limit.


class GroundingQuery(BaseModel):
    reasoning: str
    grounded: bool


_GROUNDING_QUERY_JSON_SCHEMA = GroundingQuery.model_json_schema()
_GROUNDING_QUERY_JSON_SCHEMA_STR = json.dumps(
    _GROUNDING_QUERY_JSON_SCHEMA, indent=JSON_INDENT
)
