import logging

import numpy as np

_LOG_COUNT = 5

logger = logging.getLogger(__name__)


def rank_by_rrf(
    docs: list[str],
    doc_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    inclusion_keywords: list[str],
    exclusion_keywords: list[str],
) -> list[int]:
    """Return row indices sorted by descending RRF score."""
    # Compute different scores for each row
    inclusion_scores: list[float] = [
        sum(1 for kw in inclusion_keywords if kw.lower() in doc.lower()) for doc in docs
    ]
    exclusion_scores: list[float] = [
        sum(1 for kw in exclusion_keywords if kw.lower() in doc.lower()) for doc in docs
    ]
    # Embeddings are already normalized to unit length, so dot product
    # is cosine similarity
    similarity_scores = (doc_embeddings @ query_embedding).tolist()

    # Compute ranks
    inclusion_rank = _compute_ranks(inclusion_scores, descending=True)
    exclusion_rank = _compute_ranks(exclusion_scores, descending=False)
    embedding_rank = _compute_ranks(similarity_scores, descending=True)

    # Compute RRF scores
    k = 60  # Use default value of k=60, which is the common choice for RRF
    rrf_scores = [
        1 / (k + inclusion_rank[i])
        + 1 / (k + exclusion_rank[i])
        + 1 / (k + embedding_rank[i])
        for i in range(len(docs))
    ]

    # Sort by RRF score (descending)
    sorted_indices = sorted(range(len(docs)), key=lambda i: rrf_scores[i], reverse=True)

    for rank, i in enumerate(sorted_indices[:_LOG_COUNT]):
        logger.debug(
            "TOP relevance scores (row %d): text=%s, "
            "inclusion_score=%d (rank %d), exclusion_score=%d (rank %d), "
            "similarity_score=%.4f (rank %d), rrf=%.4f (rank %d)",
            i,
            docs[i].replace("\n", " "),
            inclusion_scores[i],
            inclusion_rank[i],
            exclusion_scores[i],
            exclusion_rank[i],
            similarity_scores[i],
            embedding_rank[i],
            rrf_scores[i],
            rank,
        )

    for offset, i in enumerate(sorted_indices[-_LOG_COUNT:]):
        rank = len(sorted_indices) - _LOG_COUNT + offset
        logger.debug(
            "BOTTOM relevance scores (row %d): text=%s, "
            "inclusion_score=%d (rank %d), exclusion_score=%d (rank %d), "
            "similarity_score=%.4f (rank %d), rrf=%.4f (rank %d)",
            i,
            docs[i].replace("\n", " "),
            inclusion_scores[i],
            inclusion_rank[i],
            exclusion_scores[i],
            exclusion_rank[i],
            similarity_scores[i],
            embedding_rank[i],
            rrf_scores[i],
            rank,
        )

    return sorted_indices


def _compute_ranks(scores: list[float], descending: bool) -> dict[int, int]:
    """Convert scores to 1-indexed dense ranks (ties get same rank)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=descending)
    ranks: dict[int, int] = {}
    prev_score = None
    rank = 0
    for index in order:
        if scores[index] != prev_score:
            rank += 1
            prev_score = scores[index]
        ranks[index] = rank
    return ranks
