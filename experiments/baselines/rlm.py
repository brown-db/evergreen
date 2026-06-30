import json
import os
import time
import uuid
from typing import cast

import dspy  # type: ignore

from evergreen.catalog.schema import Schema
from evergreen.common.constants import CACHE_DIR_ROOT
from evergreen.model.config import CortexModelConfig
from evergreen.model.language_model import LanguageModelMetrics
from experiments.baselines.rag_agent import RetrievalEngine
from experiments.common import (
    CONNECTION_NAME,
    DEFAULT_LANGUAGE_MODEL,
    EMBEDDING_MODEL,
    load_snowflake_connection,
)
from experiments.types import EvaluationResult, QueryMetrics


class ClaimVerifier(dspy.Signature):
    """
    Determine whether a claim is grounded in a dataset, i.e., fully supported
    by evidence in the dataset. A claim is NOT grounded if the dataset contradicts it or
    lacks sufficient evidence. Leverage your knowledge of first-order logic and its
    extensions to interpret the claim. Identify constants, variables, predicates,
    functions, quantifiers, etc. If hints are provided, use them for clarification.
    Use the retrieve tool to quickly search the dataset for witnesses or counterexamples
    to the claim. Use the sub-LLM to semantically reason about (e.g., classify) each
    text field in the dataset. Use Python code to symbolically reason (e.g., compute
    aggregates) over the dataset. These tools are complementary and should be used in
    conjunction with each other when appropriate. Do not stop until you are absolutely
    confident in your verdict.
    """

    dataset: list[dict[str, object]] = dspy.InputField(  # type: ignore
        description="The dataset to verify the claim against"
    )
    agg_prompt: str = dspy.InputField(  # type: ignore
        description=(
            "The prompt over the dataset that generated a response containing the claim"
        )
    )
    dataset_schema: str = dspy.InputField(  # type: ignore
        description="The schema of the dataset"
    )
    claim: str = dspy.InputField(description="The claim to verify")  # type: ignore
    hints: str = dspy.InputField(  # type: ignore
        description="Optional clarifying hints for interpreting the claim"
    )
    grounded: bool = dspy.OutputField(  # type: ignore
        description=(
            "True if the claim is grounded in the dataset, i.e., fully supported by "
            "evidence; False if the dataset contradicts it or lacks sufficient evidence"
        )
    )


def evaluate_claim(
    claim: str,
    hints: str,
    agg_prompt: str,
    dataset_path: str,
    schema: Schema,
    text_field_name: str,
    language_model: str,
) -> EvaluationResult:
    conn = load_snowflake_connection(CONNECTION_NAME)
    api_base = f"https://{conn['account']}.snowflakecomputing.com/api/v2/cortex"
    assert "ANTHROPIC_API_KEY" not in os.environ, (
        "ANTHROPIC_API_KEY is set; litellm would send it as `x-api-key` and ignore "
        "the Cortex bearer PAT (ANTHROPIC_AUTH_TOKEN). Unset it to use Cortex."
    )
    os.environ["ANTHROPIC_AUTH_TOKEN"] = conn["password"]

    root_lm = dspy.LM(
        f"anthropic/{DEFAULT_LANGUAGE_MODEL}",
        api_base=api_base,
        temperature=1.0,
        max_tokens=16_384,
        cache=False,
    )
    dspy.configure(lm=root_lm, track_usage=True)  # type: ignore

    with open(dataset_path) as f:
        dataset = [
            {k: v for k, v in json.loads(line).items() if k in schema.field_names()}
            for line in f
        ]

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
        schema,
        text_field_name,
        embedding_model,
    )

    sub_lm = dspy.LM(
        f"anthropic/{language_model}",
        api_base=api_base,
        temperature=0.0,
        max_tokens=16_384,
        cache=False,
    )
    rlm = dspy.RLM(  # type: ignore
        ClaimVerifier,
        max_iterations=100,
        max_llm_calls=5_000,
        sub_lm=sub_lm,
        tools=[retrieval_engine.retrieve],
        verbose=True,
    )

    t0 = time.perf_counter()

    result = rlm(
        dataset=dataset,
        agg_prompt=agg_prompt,
        dataset_schema=str(schema),
        claim=claim,
        hints=hints,
    )

    execution_latency = time.perf_counter() - t0

    verification_result = bool(result.grounded)  # type: ignore

    reasoning = cast(list[dict[str, object]], result.trajectory)  # type: ignore

    def _get_usage(lm: dspy.LM) -> tuple[int, int, int]:
        history = cast(list[dict[str, object]], lm.history)  # type: ignore
        prompt_count = len(history)
        input_tokens = 0
        output_tokens = 0
        for entry in history:
            usage = cast(dict[str, int], entry["usage"])
            input_tokens += usage["prompt_tokens"]
            output_tokens += usage["completion_tokens"]
        return prompt_count, input_tokens, output_tokens

    root_prompt_count, root_input_tokens, root_output_tokens = _get_usage(root_lm)
    sub_prompt_count, sub_input_tokens, sub_output_tokens = _get_usage(sub_lm)

    lm_metrics = (
        LanguageModelMetrics(
            model_name=language_model,
            prompt_count=sub_prompt_count,
            input_token_count=sub_input_tokens,
            output_token_count=sub_output_tokens,
        ),
    )
    optimizer_lm_metrics = (
        LanguageModelMetrics(
            model_name=DEFAULT_LANGUAGE_MODEL,
            prompt_count=root_prompt_count,
            input_token_count=root_input_tokens,
            output_token_count=root_output_tokens,
        ),
    )

    return EvaluationResult(
        verification_result=verification_result,
        query_metrics=QueryMetrics(
            planning_latency=0.0,
            execution_latency=execution_latency,
            language_model_metrics=lm_metrics,
            embedding_model_metrics=embedding_model.metrics(),
            optimizer_language_model_metrics=optimizer_lm_metrics,
        ),
        filter_metrics=None,
        map_metrics=None,
        prov_tokens={},
        reasoning=reasoning,
    )
