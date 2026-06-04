from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import cast

from nltk.tokenize import sent_tokenize  # type: ignore
from snowflake.snowpark import Session
from snowflake.snowpark.functions import ai_embed, col

from evergreen.common.cache import Cache, FileCache

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModelMetrics:
    model_name: str
    embed_count: int = 0
    embed_cache_hit_count: int = 0

    def copy(self) -> EmbeddingModelMetrics:
        return EmbeddingModelMetrics(
            self.model_name,
            self.embed_count,
            self.embed_cache_hit_count,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "embed_count": self.embed_count,
            "embed_cache_hit_count": self.embed_cache_hit_count,
        }

    def clear(self) -> None:
        self.embed_count = 0
        self.embed_cache_hit_count = 0


class EmbeddingType(Enum):
    QUERY = auto()
    DOCUMENT = auto()


class EmbeddingModel(ABC):
    def __init__(self, model_name: str, cache_dir: Path | None):
        self._model_name = model_name
        self._cache: Cache | None = (
            FileCache(cache_dir) if cache_dir is not None else None
        )

    @abstractmethod
    def metrics(self) -> tuple[EmbeddingModelMetrics, ...]:
        pass

    @abstractmethod
    def clear_metrics(self) -> None:
        pass

    @abstractmethod
    def embed(self, text: str, embedding_type: EmbeddingType) -> list[float]:
        pass

    @abstractmethod
    def embed_batch(
        self, texts: list[str], embedding_type: EmbeddingType
    ) -> list[list[float]]:
        pass

    def embed_with_sentences_batch(
        self, texts: list[str]
    ) -> list[tuple[list[float], list[list[float]]]]:
        if not texts:
            return []

        doc_embeddings = self.embed_batch(texts, EmbeddingType.DOCUMENT)

        sentences_per_text = [sent_tokenize(text) for text in texts]
        sentence_embeddings = self.embed_batch(
            [sentence for sentences in sentences_per_text for sentence in sentences],
            EmbeddingType.DOCUMENT,
        )

        results: list[tuple[list[float], list[list[float]]]] = []
        offset = 0
        for doc_embedding, sentences in zip(
            doc_embeddings, sentences_per_text, strict=True
        ):
            next_offset = offset + len(sentences)
            results.append((doc_embedding, sentence_embeddings[offset:next_offset]))
            offset = next_offset

        return results

    def _cache_key(self, text: str) -> str:
        key_data = json.dumps(
            {
                "model_name": self._model_name,
                "text": text,
            },
            sort_keys=True,
        )
        return hashlib.md5(key_data.encode()).hexdigest()


class CortexEmbeddingModel(EmbeddingModel):
    _PREFIX = {
        "snowflake-arctic-embed-l-v2.0": {
            EmbeddingType.QUERY: "query: ",
            EmbeddingType.DOCUMENT: "",
        },
    }

    def __init__(
        self,
        model_name: str,
        cache_dir: Path | None,
        session: Session,
    ):
        super().__init__(model_name, cache_dir)
        self._session = session
        self._metrics = EmbeddingModelMetrics(model_name)

    def metrics(self) -> tuple[EmbeddingModelMetrics, ...]:
        return (self._metrics.copy(),)

    def clear_metrics(self) -> None:
        self._metrics.clear()

    def embed(self, text: str, embedding_type: EmbeddingType) -> list[float]:
        self._metrics.embed_count += 1

        text = f"{self._PREFIX[self._model_name][embedding_type]}{text}"

        cache_key = self._cache_key(text)

        if self._cache is not None and cache_key in self._cache:
            logger.debug("%s cache hit (%s): %s", self._model_name, cache_key, text)
            self._metrics.embed_cache_hit_count += 1
            return cast(list[float], self._cache[cache_key])

        logger.debug("%s embed: %s", self._model_name, text)

        df = self._session.range(1).select(ai_embed(self._model_name, text))

        embedding = cast(list[float], df.collect()[0][0])

        if self._cache is not None:
            self._cache[cache_key] = embedding

        return embedding

    def embed_batch(
        self, texts: list[str], embedding_type: EmbeddingType
    ) -> list[list[float]]:
        if not texts:
            return []

        texts = [
            f"{self._PREFIX[self._model_name][embedding_type]}{text}" for text in texts
        ]

        embeddings: list[list[float] | None] = []
        uncached_texts: list[tuple[int, str]] = []  # list of (index, text)

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)

            if self._cache is not None and cache_key in self._cache:
                embeddings.append(cast(list[float], self._cache[cache_key]))
            else:
                embeddings.append(None)
                uncached_texts.append((i, text))

        if uncached_texts:
            rows = (
                self._session.create_dataframe(  # type: ignore
                    uncached_texts,
                    schema=["INDEX", "TEXT"],
                )
                .select(
                    col("INDEX"),
                    ai_embed(self._model_name, col("TEXT")).alias("EMBEDDING"),
                )
                .order_by(col("INDEX"))
                .collect()
            )
            uncached_embeddings = [cast(list[float], row["EMBEDDING"]) for row in rows]

            for (i, text), embedding in zip(
                uncached_texts, uncached_embeddings, strict=True
            ):
                embeddings[i] = embedding
                if self._cache is not None:
                    self._cache[self._cache_key(text)] = embedding

        self._metrics.embed_count += len(texts)
        self._metrics.embed_cache_hit_count += len(texts) - len(uncached_texts)

        return cast(list[list[float]], embeddings)
