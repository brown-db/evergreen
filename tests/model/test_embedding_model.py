import uuid

import numpy as np

from evergreen.common.constants import CACHE_DIR_ROOT
from evergreen.model.config import CortexModelConfig
from evergreen.model.embedding_model import EmbeddingType
from tests.conftest import (
    DEFAULT_CONNECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LANGUAGE_MODEL,
)


class TestCortexEmbeddingModel:
    def test_embed(self):
        model_config = CortexModelConfig(
            [DEFAULT_LANGUAGE_MODEL], DEFAULT_EMBEDDING_MODEL, DEFAULT_CONNECTION_NAME
        )
        model = model_config.create_embedding_model(
            cache_dir=CACHE_DIR_ROOT / uuid.uuid4().hex
        )
        texts: list[str] = [
            "This is a test sentence.",
            "This is another test sentence.",
            "This is a test sentence.",  # duplicate
        ]
        embeddings = [model.embed(text, EmbeddingType.QUERY) for text in texts]

        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            assert np.isclose(norm, 1.0)
            assert len(embedding) == 1024

        assert len(model.metrics()) == 1
        assert model.metrics()[0].embed_count == 3
        assert model.metrics()[0].embed_cache_hit_count == 1

    def test_embed_batch(self):
        model_config = CortexModelConfig(
            [DEFAULT_LANGUAGE_MODEL], DEFAULT_EMBEDDING_MODEL, DEFAULT_CONNECTION_NAME
        )
        model = model_config.create_embedding_model(
            cache_dir=CACHE_DIR_ROOT / uuid.uuid4().hex
        )
        texts: list[str] = [
            "This is a test sentence.",
            "This is another test sentence.",
            "This is a test sentence.",  # duplicate
        ]
        embeddings = model.embed_batch(texts, EmbeddingType.DOCUMENT)

        assert len(embeddings) == 3
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            assert np.isclose(norm, 1.0)
            assert len(embedding) == 1024

        # First two are unique, third is duplicate;
        # however, due to batching, all requests occur at around the same time,
        # so we expect no cache hits
        assert model.metrics()[0].embed_count == 3
        assert model.metrics()[0].embed_cache_hit_count == 0

    def test_embed_with_sentences_batch(self):
        model_config = CortexModelConfig(
            [DEFAULT_LANGUAGE_MODEL], DEFAULT_EMBEDDING_MODEL, DEFAULT_CONNECTION_NAME
        )
        model = model_config.create_embedding_model(
            cache_dir=CACHE_DIR_ROOT / uuid.uuid4().hex
        )
        texts = [
            "This is the first sentence. This is the second sentence.",
            "Only one sentence here.",
            "First. Second. Third.",
        ]
        expected_sentence_counts = [2, 1, 3]

        results = model.embed_with_sentences_batch(texts)

        assert len(results) == len(texts)
        for (doc_embedding, sentence_embeddings), expected_count in zip(
            results, expected_sentence_counts, strict=True
        ):
            # Doc embedding
            assert len(doc_embedding) == 1024
            assert np.isclose(np.linalg.norm(doc_embedding), 1.0)

            # Sentence embeddings regrouped to the correct text
            assert len(sentence_embeddings) == expected_count
            for sentence_embedding in sentence_embeddings:
                assert len(sentence_embedding) == 1024
                assert np.isclose(np.linalg.norm(sentence_embedding), 1.0)

        # 3 doc embeds + (2 + 1 + 3) sentence embeds = 9 total
        assert model.metrics()[0].embed_count == 9

    def test_embed_with_sentences_batch_empty_input(self):
        model_config = CortexModelConfig(
            [DEFAULT_LANGUAGE_MODEL], DEFAULT_EMBEDDING_MODEL, DEFAULT_CONNECTION_NAME
        )
        model = model_config.create_embedding_model(
            cache_dir=CACHE_DIR_ROOT / uuid.uuid4().hex
        )

        assert model.embed_with_sentences_batch([]) == []
        assert model.metrics()[0].embed_count == 0

    def test_embed_with_sentences_batch_empty_string(self):
        model_config = CortexModelConfig(
            [DEFAULT_LANGUAGE_MODEL], DEFAULT_EMBEDDING_MODEL, DEFAULT_CONNECTION_NAME
        )
        model = model_config.create_embedding_model(
            cache_dir=CACHE_DIR_ROOT / uuid.uuid4().hex
        )
        # An empty string contributes zero sentences; it must not shift the
        # offsets of subsequent texts.
        texts = ["A sentence here.", "", "First. Second."]
        expected_sentence_counts = [1, 0, 2]

        results = model.embed_with_sentences_batch(texts)

        assert len(results) == len(texts)
        for (doc_embedding, sentence_embeddings), expected_count in zip(
            results, expected_sentence_counts, strict=True
        ):
            # Doc embedding exists even for the empty string
            assert len(doc_embedding) == 1024
            assert len(sentence_embeddings) == expected_count
