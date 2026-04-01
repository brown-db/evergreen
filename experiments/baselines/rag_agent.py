import json
import logging
import time
import uuid
from typing import Any, cast

import httpx
import numpy as np
from anthropic import Anthropic, Omit, omit
from anthropic.types import (
    ContentBlock,
    MessageParam,
    OutputConfigParam,
    TextBlockParam,
    ThinkingConfigParam,
    ToolParam,
)
from pydantic import BaseModel, Field, create_model

from evergreen.catalog.schema import Field as SchemaField
from evergreen.common.constants import CACHE_DIR_ROOT, EMBEDDING_FIELD_SUFFIX
from evergreen.core.session_context import SessionContext
from evergreen.data_frame import QueryMetrics
from evergreen.model.config import CortexModelConfig
from evergreen.model.embedding_model import EmbeddingModel, EmbeddingType
from evergreen.model.language_model import LanguageModelMetrics
from evergreen.retrieval import rank_by_rrf
from experiments.common import (
    CONNECTION_NAME,
    EMBEDDING_MODEL,
    load_snowflake_connection,
)
from experiments.types import EvaluationResult

logger = logging.getLogger(__name__)


class RetrievalEngine:
    def __init__(
        self,
        dataset_path: str,
        dataset_key: tuple[str, ...],
        text_field_name: str,
        schema_field_names: set[str],
        embedding_model: EmbeddingModel,
    ) -> None:
        df = SessionContext().read_json(
            dataset_path,
            dataset_key,
        )

        self._schema = df.schema()
        self._schema_field_names = schema_field_names

        text_field_index = self._schema.index_of(text_field_name)
        embedding_index = self._schema.index_of(
            text_field_name + EMBEDDING_FIELD_SUFFIX
        )
        field_indices = {
            name: self._schema.index_of(name) for name in self._schema_field_names
        }

        doc_embeddings: list[list[float]] = []

        self._docs: list[str] = []
        self._row_strs: list[str] = []
        self._rows: list[dict[str, object]] = []

        for row in df.collect().rows:
            doc_embeddings.append(cast(list[float], row[embedding_index]))
            filtered_obj = {
                name: row[field_indices[name]] for name in self._schema_field_names
            }
            self._docs.append(str(row[text_field_index]))
            self._row_strs.append(json.dumps(filtered_obj))
            self._rows.append(filtered_obj)

        self._doc_embeddings = np.array(doc_embeddings)

        self._embedding_model = embedding_model

    @property
    def row_count(self) -> int:
        return len(self._row_strs)

    @property
    def schema_fields(self) -> tuple[SchemaField, ...]:
        return tuple(
            field
            for field in self._schema.fields
            if field.name in self._schema_field_names
        )

    def retrieve(
        self,
        query_text: str,
        inclusion_keywords: list[str],
        exclusion_keywords: list[str],
        filters: dict[str, object],
        k: int,
    ) -> list[str]:
        filters = {k: v for k, v in filters.items() if v is not None}

        if filters:
            candidate_indices = [
                i
                for i, row in enumerate(self._rows)
                if all(row.get(key) == value for key, value in filters.items())
            ]
        else:
            candidate_indices = list(range(len(self._rows)))

        filtered_docs = [self._docs[i] for i in candidate_indices]
        filtered_doc_embeddings = self._doc_embeddings[candidate_indices]

        query_embedding = np.array(
            self._embedding_model.embed(query_text, EmbeddingType.QUERY)
        )

        sorted_indices = rank_by_rrf(
            filtered_docs,
            filtered_doc_embeddings,
            query_embedding,
            inclusion_keywords,
            exclusion_keywords,
        )

        return [self._row_strs[candidate_indices[i]] for i in sorted_indices[:k]]


class RetrieveParams(BaseModel):
    query_text: str = Field(
        description="Semantic search query used to find relevant rows "
        "via embedding similarity. Choose a query that captures the "
        "meaning of the evidence you need."
    )
    inclusion_keywords: list[str] = Field(
        description="Keywords/phrases that boost ranking of rows "
        "containing them (soft signal, not a hard filter). "
        "Matched as case-insensitive substrings, so prefer shorter, "
        "atomic terms. Include common variations and abbreviations."
    )
    exclusion_keywords: list[str] = Field(
        description="Keywords/phrases that penalize ranking of rows "
        "containing them (soft signal, not a hard filter). "
        "Matched as case-insensitive substrings, so prefer shorter, "
        "atomic terms."
    )
    k: int = Field(
        description=(
            "Number of rows to retrieve. Can be up to the total "
            "row count for exhaustive search. Only the first "
            "page_size rows are returned immediately; use "
            "continue_reading to view the rest."
        )
    )
    page_size: int = Field(description="Number of retrieved rows to view immediately.")


class ContinueReadingParams(BaseModel):
    page_size: int = Field(
        description="Number of rows to read next. Picks up from where "
        "the last page ended. Calling retrieve again resets the cursor."
    )


_CONTINUE_READING_SCHEMA = ContinueReadingParams.model_json_schema()
_CONTINUE_READING_SCHEMA["additionalProperties"] = False


class RespondParams(BaseModel):
    grounded: bool = Field(
        description="true if the claim is grounded in the dataset, "
        "i.e., fully supported by evidence; false if the dataset "
        "contradicts it or lacks sufficient evidence."
    )


_RESPOND_SCHEMA = RespondParams.model_json_schema()
_RESPOND_SCHEMA["additionalProperties"] = False


def _build_tools(
    schema_fields: tuple[SchemaField, ...], text_field_name: str
) -> list[ToolParam]:
    # Dynamic filter model from dataset schema
    filter_fields: dict[str, Any] = {
        field.name: (field.dtype | None)
        for field in schema_fields
        if field.name != text_field_name
    }
    FilterParams = create_model("FilterParams", **filter_fields)

    # Build retrieve schema with dynamic filters
    retrieve_schema = RetrieveParams.model_json_schema()
    retrieve_schema["additionalProperties"] = False
    filter_schema = FilterParams.model_json_schema()
    filter_schema["additionalProperties"] = False
    retrieve_schema["properties"]["filters"] = {
        **filter_schema,
        "description": (
            "Hard equality filters on non-text fields. "
            "Only rows matching all non-null filters are returned. "
            "Leave fields as null for no filtering."
        ),
    }
    retrieve_schema["required"].append("filters")

    return [
        {
            "name": "retrieve",
            "strict": True,
            "description": (
                "Run a new search over the full dataset. Results are ranked by "
                "a combination of semantic similarity (query_text) and keyword "
                "matching (inclusion/exclusion_keywords) using Reciprocal Rank "
                "Fusion. Returns the first page_size rows; use continue_reading "
                "to paginate through the rest. Calling retrieve again replaces "
                "the previous result set and resets the cursor."
            ),
            "input_schema": retrieve_schema,
        },
        {
            "name": "continue_reading",
            "strict": True,
            "description": (
                "Return the next page of rows from the most recent retrieve "
                "call, starting where the last page ended. Use this to "
                "paginate through a large result set without re-running the "
                "search. Returns an empty page if all rows have already been read."
            ),
            "input_schema": _CONTINUE_READING_SCHEMA,
        },
        {
            "name": "respond",
            "strict": True,
            "description": (
                "Submit your final verdict. Call this once you have gathered "
                "enough evidence to determine whether the claim is grounded. "
                "This ends the evaluation."
            ),
            "input_schema": _RESPOND_SCHEMA,
        },
    ]


def _format_page(rows: list[str], cursor: int, page_size: int) -> tuple[str, int]:
    page = rows[cursor : cursor + page_size]
    new_cursor = cursor + len(page)
    remaining = len(rows) - new_cursor
    text = "\n".join(page) + f"\n[ROWS_SHOWN={len(page)} ROWS_REMAINING={remaining}]"
    logger.debug("tool_result:\n%s", text)
    return text, new_cursor


def _add_cache_control(messages: list[MessageParam]) -> list[MessageParam]:
    *head, last = messages
    content = last["content"]
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    blocks = cast(list[ContentBlock], content)
    last_block = cast(dict[str, object], blocks[-1])
    return [
        *head,
        cast(
            MessageParam,
            {
                "role": "user",
                "content": [
                    *blocks[:-1],
                    {**last_block, "cache_control": {"type": "ephemeral"}},
                ],
            },
        ),
    ]


_SYSTEM_PROMPT_TEMPLATE = """\
You are a fact-checking agent. Your goal is to determine whether a claim is \
grounded in a dataset, i.e., fully supported by evidence in the dataset.
A claim is NOT grounded if the dataset contradicts it or lacks sufficient evidence.

## Dataset
- {row_count} total rows.
- Fields per row: {field_names}.
- Primary text field: `{text_field_name}`.

## Reasoning strategy
At each step, think carefully about:
1. **Claim formulation** — formally express the logical structure of the claim. \
Leverage your knowledge of first-order logic and its extensions. Identify \
constants, variables, predicates, functions, quantifiers, etc. If hints are \
provided, use them for clarification on vague quantifier thresholds.
2. **Progress review** — summarize what you have searched for and found so far.
3. **Gap analysis** — identify what evidence is still missing.
4. **Next action** — decide whether to retrieve, continue_reading, or respond.

You are encouraged to set k to a large value (even to the total row count) for \
exhaustive search and then paginate with continue_reading in order to gather \
necessary evidence. Do not stop until you are absolutely confident in your verdict."""


def evaluate_claim(
    claim: str,
    hints: str,
    agg_prompt: str,
    dataset_path: str,
    dataset_key: tuple[str, ...],
    text_field_name: str,
    schema_field_names: set[str],
    language_model: str,
) -> EvaluationResult:
    conn = load_snowflake_connection(CONNECTION_NAME)
    pat = conn["password"]
    client = Anthropic(
        base_url=f"https://{conn['account']}.snowflakecomputing.com/api/v2/cortex",
        http_client=httpx.Client(headers={"Authorization": f"Bearer {pat}"}),
        default_headers={"Authorization": f"Bearer {pat}"},
    )

    model_config = CortexModelConfig(
        language_models=[language_model],
        embedding_model=EMBEDDING_MODEL,
        connection_name=CONNECTION_NAME,
    )
    embedding_model = model_config.create_embedding_model(
        cache_dir=CACHE_DIR_ROOT / uuid.uuid4().hex
    )
    retrieval_engine = RetrievalEngine(
        dataset_path,
        dataset_key,
        text_field_name,
        schema_field_names,
        embedding_model,
    )

    tools = _build_tools(retrieval_engine.schema_fields, text_field_name)

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        row_count=retrieval_engine.row_count,
        field_names=", ".join(f"`{name}`" for name in sorted(schema_field_names)),
        text_field_name=text_field_name,
    )

    system: list[TextBlockParam] = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "The claim below was in response to the following prompt: "
                    f"{agg_prompt}\n\n"
                    f"<claim>{claim}</claim>\n<hints>{hints}</hints>",
                }
            ],
        },
    ]

    thinking_param: ThinkingConfigParam | Omit = (
        {"type": "adaptive"}
        if language_model in {"claude-opus-4-6", "claude-sonnet-4-6"}
        else omit
    )

    output_config: OutputConfigParam | Omit = (
        {"effort": "high"}
        if language_model in {"claude-opus-4-6", "claude-sonnet-4-6"}
        else omit
    )

    outputs: list[dict[str, object]] = []
    retrieved_rows: list[str] = []
    cursor = 0
    step = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation_input_tokens = 0
    total_cache_read_input_tokens = 0

    t0 = time.perf_counter()

    while True:
        raw_response = client.messages.with_raw_response.create(
            model=language_model,
            max_tokens=16_384,
            system=system,
            messages=_add_cache_control(messages),
            thinking=thinking_param,
            output_config=output_config,
            temperature=1.0,
            tools=tools,
            tool_choice={"type": "auto", "disable_parallel_tool_use": True},
        )
        response = raw_response.parse()
        request_id = raw_response.headers["x-snowflake-request-id"]

        thinking = ""
        text = ""
        tool_use = None
        for block in response.content:
            if block.type == "thinking":
                thinking = block.thinking
            elif block.type == "text":
                text = block.text
            elif block.type == "tool_use":
                tool_use = block

        usage = response.usage
        total_input_tokens += usage.input_tokens
        total_output_tokens += usage.output_tokens
        total_cache_creation_input_tokens += usage.cache_creation_input_tokens or 0
        total_cache_read_input_tokens += usage.cache_read_input_tokens or 0

        messages.append({"role": "assistant", "content": response.content})

        step_output: dict[str, object] = {
            "step": step,
            "request_id": request_id,
            "thinking": thinking,
            "text": text,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cache_creation_input_tokens": usage.cache_creation_input_tokens,
                "cache_read_input_tokens": usage.cache_read_input_tokens,
            },
        }
        if tool_use is not None:
            step_output["tool"] = tool_use.name
            step_output["arguments"] = dict(tool_use.input)
        outputs.append(step_output)

        if tool_use is None:
            logger.debug(
                "%s step %d: request_id=%s, no tool call, thinking=%s, text=%s, "
                "tokens(in=%s, out=%s, cache_creation_in=%s, cache_read_in=%s)",
                language_model,
                step,
                request_id,
                thinking,
                text,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_creation_input_tokens,
                usage.cache_read_input_tokens,
            )
            messages.append(
                {
                    "role": "user",
                    "content": "Please use a tool to continue your investigation or "
                    "call respond with your verdict.",
                }
            )
            step += 1
            continue

        logger.debug(
            "%s step %d: request_id=%s, thinking=%s, text=%s, tool=%s, args=%s, "
            "tokens(in=%s, out=%s, cache_creation_in=%s, cache_read_in=%s)",
            language_model,
            step,
            request_id,
            thinking,
            text,
            tool_use.name,
            tool_use.input,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_creation_input_tokens,
            usage.cache_read_input_tokens,
        )

        match tool_use.name:
            case "retrieve":
                params = RetrieveParams.model_validate(tool_use.input)
                filters = cast(dict[str, object], tool_use.input["filters"])
                retrieved_rows = retrieval_engine.retrieve(
                    params.query_text,
                    params.inclusion_keywords,
                    params.exclusion_keywords,
                    filters,
                    params.k,
                )
                cursor = 0
                tool_result, cursor = _format_page(
                    retrieved_rows, cursor, params.page_size
                )
            case "continue_reading":
                params = ContinueReadingParams.model_validate(tool_use.input)
                tool_result, cursor = _format_page(
                    retrieved_rows, cursor, params.page_size
                )
            case "respond":
                params = RespondParams.model_validate(tool_use.input)
                lm_metrics = LanguageModelMetrics(
                    model_name=language_model,
                    prompt_count=step + 1,
                    input_token_count=total_input_tokens,
                    output_token_count=total_output_tokens,
                    cache_creation_input_token_count=total_cache_creation_input_tokens,
                    cache_read_input_token_count=total_cache_read_input_tokens,
                )
                return EvaluationResult(
                    verification_result=params.grounded,
                    query_metrics=QueryMetrics(
                        planning_latency=0.0,
                        execution_latency=time.perf_counter() - t0,
                        language_model_metrics=(lm_metrics,),
                        embedding_model_metrics=embedding_model.metrics(),
                        optimizer_language_model_metrics=(),
                    ),
                    filter_metrics=None,
                    map_metrics=None,
                    prov_tokens={},
                    reasoning=outputs,
                )
            case _:
                raise ValueError(f"Unknown tool: {tool_use.name}")

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": tool_result,
                    }
                ],
            }
        )
        step += 1
