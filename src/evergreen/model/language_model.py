from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, create_model
from snowflake.snowpark import Session
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.snowpark.functions import ai_complete, col  # type: ignore
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
)

from evergreen.common.cache import Cache, FileCache

logger = logging.getLogger(__name__)


def create_output_schema(
    return_type: type[object] | tuple[type[object], ...],
) -> type[BaseModel]:
    if isinstance(return_type, tuple):
        fields: dict[str, Any] = {
            f"output_{i}": rtype for i, rtype in enumerate(return_type)
        }
        return create_model("OutputSchema", **fields)
    return create_model("OutputSchema", output=return_type)


@dataclass
class LanguageModelMetrics:
    model_name: str
    prompt_count: int = 0
    prompt_cache_hit_count: int = 0
    input_token_count: int = 0
    output_token_count: int = 0
    cache_creation_input_token_count: int = 0
    cache_read_input_token_count: int = 0

    def copy(self) -> LanguageModelMetrics:
        return LanguageModelMetrics(
            self.model_name,
            self.prompt_count,
            self.prompt_cache_hit_count,
            self.input_token_count,
            self.output_token_count,
            self.cache_creation_input_token_count,
            self.cache_read_input_token_count,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "prompt_count": self.prompt_count,
            "prompt_cache_hit_count": self.prompt_cache_hit_count,
            "input_token_count": self.input_token_count,
            "output_token_count": self.output_token_count,
            "cache_creation_input_token_count": self.cache_creation_input_token_count,
            "cache_read_input_token_count": self.cache_read_input_token_count,
        }

    def clear(self) -> None:
        self.prompt_count = 0
        self.prompt_cache_hit_count = 0
        self.input_token_count = 0
        self.output_token_count = 0
        self.cache_creation_input_token_count = 0
        self.cache_read_input_token_count = 0


class LanguageModel(ABC):
    def __init__(self, model_name: str, cache_dir: Path | None):
        self._model_name = model_name
        self._cache: Cache | None = (
            FileCache(cache_dir) if cache_dir is not None else None
        )

    @abstractmethod
    def metrics(self) -> tuple[LanguageModelMetrics, ...]:
        pass

    @abstractmethod
    def clear_metrics(self) -> None:
        pass

    @abstractmethod
    def prompt[T](
        self,
        prompt_strs: list[str],
        return_type: type[T],
        schema: type[BaseModel] | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> list[T]:
        pass

    @abstractmethod
    def prompt_with_schema[T: BaseModel](
        self, prompt_str: str, schema: type[T], json_schema: dict[str, object]
    ) -> T:
        pass

    @abstractmethod
    def prompt_without_schema(self, prompt_str: str) -> str:
        pass

    @abstractmethod
    def fused_prompt(
        self,
        prompt_strs: list[str],
        return_types: tuple[type[object], ...],
        schema: type[BaseModel] | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> list[tuple[object, ...]]:
        pass

    def _cache_key(self, prompt_str: str) -> str:
        key_data = json.dumps(
            {
                "model_name": self._model_name,
                "prompt_str": prompt_str,
            },
            sort_keys=True,
        )
        return hashlib.md5(key_data.encode()).hexdigest()


class CortexLanguageModel(LanguageModel):
    MAX_OUTPUT_TOKENS = 8192

    def __init__(
        self,
        model_name: str,
        cache_dir: Path | None,
        session: Session,
    ):
        super().__init__(model_name, cache_dir)
        self._session = session
        self._metrics = LanguageModelMetrics(model_name)

    def metrics(self) -> tuple[LanguageModelMetrics, ...]:
        return (self._metrics.copy(),)

    def clear_metrics(self) -> None:
        self._metrics.clear()

    def prompt[T](
        self,
        prompt_strs: list[str],
        return_type: type[T],
        schema: type[BaseModel] | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> list[T]:
        if schema is None:
            schema = create_output_schema(return_type)
        if json_schema is None:
            json_schema = schema.model_json_schema()
        responses = self._complete(prompt_strs, schema, json_schema)
        return [response.output for response in responses]  # type: ignore

    def prompt_with_schema[T: BaseModel](
        self, prompt_str: str, schema: type[T], json_schema: dict[str, object]
    ) -> T:
        return cast(T, self._complete([prompt_str], schema, json_schema)[0])

    def fused_prompt(
        self,
        prompt_strs: list[str],
        return_types: tuple[type[object], ...],
        schema: type[BaseModel] | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> list[tuple[object, ...]]:
        if schema is None:
            schema = create_output_schema(return_types)
        if json_schema is None:
            json_schema = schema.model_json_schema()
        responses = self._complete(prompt_strs, schema, json_schema)
        return [
            tuple(getattr(response, f"output_{i}") for i in range(len(return_types)))
            for response in responses
        ]

    def prompt_without_schema(self, prompt_str: str) -> str:
        self._metrics.prompt_count += 1

        cache_key, result = self._read_from_cache(prompt_str)

        if result is not None:
            return self._extract_unstructured_output(result)

        df = self._session.range(1).select(
            ai_complete(
                model=self._model_name,
                prompt=prompt_str,
                show_details=True,
            )
        )

        result = json.loads(cast(str, df.collect()[0][0]))

        logger.debug(
            "%s prompt: prompt_str=%s, result=%s", self._model_name, prompt_str, result
        )

        output = self._extract_unstructured_output(result)
        usage = cast(dict[str, int], result["usage"])

        self._metrics.input_token_count += usage["prompt_tokens"]
        self._metrics.output_token_count += usage["completion_tokens"]

        if self._cache is not None:
            self._cache[cache_key] = result

        return output

    def _complete(
        self,
        prompt_strs: list[str],
        schema: type[BaseModel],
        json_schema: dict[str, object],
    ) -> list[BaseModel]:
        self._metrics.prompt_count += len(prompt_strs)

        # OpenAI (GPT) models require additionalProperties to be False
        # https://docs.snowflake.com/en/user-guide/snowflake-cortex/complete-structured-outputs
        json_schema = {**json_schema, "additionalProperties": False}

        responses: list[BaseModel | None] = [None] * len(prompt_strs)
        cache_misses: list[tuple[int, str]] = []  # (index, cache_key)

        # Check cache for each prompt
        for i, prompt_str in enumerate(prompt_strs):
            cache_key, result = self._read_from_cache(prompt_str)
            if result is not None:
                output = self._extract_structured_output(result)
                responses[i] = schema.model_validate(output)
            else:
                cache_misses.append((i, cache_key))

        # Call API for cache misses
        if cache_misses:
            results = self._call_api(
                [prompt_strs[i] for i, _ in cache_misses], json_schema
            )

            for (i, cache_key), result in zip(cache_misses, results, strict=True):
                output = self._extract_structured_output(result)
                usage = cast(dict[str, int], result["usage"])

                self._metrics.input_token_count += usage["prompt_tokens"]
                self._metrics.output_token_count += usage["completion_tokens"]

                if self._cache is not None:
                    self._cache[cache_key] = result

                responses[i] = schema.model_validate(output)

        return cast(list[BaseModel], responses)

    @retry(
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type((KeyError, SnowparkSQLException, TypeError)),
        before_sleep=lambda retry_state: logger.warning(
            "Retry %d for %s after error: %s",
            retry_state.attempt_number,
            retry_state.fn.__name__ if retry_state.fn else "unknown",
            retry_state.outcome.exception() if retry_state.outcome else "unknown",
        ),
    )
    def _call_api(
        self, prompt_strs: list[str], json_schema: dict[str, object]
    ) -> list[dict[str, object]]:
        rows = (
            self._session.create_dataframe(  # type: ignore
                list(enumerate(prompt_strs)), schema=["INDEX", "PROMPT_STR"]
            )
            .select(
                col("INDEX"),
                ai_complete(
                    model=self._model_name,
                    prompt=col("PROMPT_STR"),
                    model_parameters={"max_tokens": self.MAX_OUTPUT_TOKENS},
                    response_format={"type": "json", "schema": json_schema},
                    show_details=True,
                ).alias("RESULT"),
            )
            .order_by(col("INDEX"))
            .collect()
        )
        results = [json.loads(cast(str, row["RESULT"])) for row in rows]

        logger.debug(
            "%s: completed batch of %d prompts", self._model_name, len(prompt_strs)
        )
        for i, (prompt_str, result) in enumerate(
            zip(prompt_strs, results, strict=True)
        ):
            logger.debug(
                "%s prompt (%d / %d): prompt_str=%s, result=%s",
                self._model_name,
                i + 1,
                len(prompt_strs),
                prompt_str,
                result,
            )

        # Validate before returning (triggers retry if incomplete)
        # Sometimes the result is malformed (i.e., no structured_output key and
        # empty usage dict)
        for result in results:
            _ = self._extract_structured_output(result)

        return results

    def _read_from_cache(self, prompt_str: str) -> tuple[str, dict[str, object] | None]:
        cache_key = self._cache_key(prompt_str)

        if self._cache is not None and cache_key in self._cache:
            self._metrics.prompt_cache_hit_count += 1
            result = cast(dict[str, object], self._cache[cache_key])
            logger.debug(
                "%s cache hit (%s): prompt_str=%s, result=%s",
                self._model_name,
                cache_key,
                prompt_str,
                result,
            )
            return cache_key, result

        return cache_key, None

    @staticmethod
    def _extract_unstructured_output(result: dict[str, object]) -> str:
        choices = cast(list[dict[str, str]], result["choices"])
        return choices[0]["messages"]

    @staticmethod
    def _extract_structured_output(result: dict[str, object]) -> dict[str, object]:
        structured_output = cast(list[dict[str, object]], result["structured_output"])
        raw_message = cast(dict[str, object], structured_output[0]["raw_message"])
        return raw_message


class EnsembleLanguageModel(LanguageModel):
    def __init__(self, models: tuple[LanguageModel, ...], cache_dir: Path | None):
        model_name = f"ensemble({', '.join([model._model_name for model in models])})"
        super().__init__(model_name, cache_dir)
        self._models = models

    def metrics(self) -> tuple[LanguageModelMetrics, ...]:
        return tuple(metrics for model in self._models for metrics in model.metrics())

    def clear_metrics(self) -> None:
        for model in self._models:
            model.clear_metrics()

    def prompt[T](
        self,
        prompt_strs: list[str],
        return_type: type[T],
        schema: type[BaseModel] | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> list[T]:
        if schema is None:
            schema = create_output_schema(return_type)
        if json_schema is None:
            json_schema = schema.model_json_schema()

        with ThreadPoolExecutor(max_workers=len(self._models)) as executor:

            def call_model(model: LanguageModel) -> list[T]:
                return model.prompt(prompt_strs, return_type, schema, json_schema)

            responses = list(executor.map(call_model, self._models))

        voted_responses: list[T] = []

        for i in range(len(prompt_strs)):
            votes = [responses[m][i] for m in range(len(self._models))]
            counter = Counter(votes)
            winner = counter.most_common(1)[0][0]

            logger.debug(
                "%s majority vote for prompt %d: %s (votes: %s)",
                self._model_name,
                i,
                winner,
                dict(counter),
            )

            voted_responses.append(winner)

        return voted_responses

    def prompt_with_schema[T: BaseModel](
        self, prompt_str: str, schema: type[T], json_schema: dict[str, object]
    ) -> T:
        raise NotImplementedError(
            "prompt_with_schema is not supported for ensemble language models"
        )

    def fused_prompt(
        self,
        prompt_strs: list[str],
        return_types: tuple[type[object], ...],
        schema: type[BaseModel] | None = None,
        json_schema: dict[str, object] | None = None,
    ) -> list[tuple[object, ...]]:
        raise NotImplementedError(
            "fused_prompt is not supported for ensemble language models"
        )

    def prompt_without_schema(self, prompt_str: str) -> str:
        raise NotImplementedError(
            "prompt_without_schema is not supported for ensemble language models"
        )
