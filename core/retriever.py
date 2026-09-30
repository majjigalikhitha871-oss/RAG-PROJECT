from dataclasses import dataclass

import numpy as np

from core.chunker import Chunk
from core.embeddings import normalize_vectors


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def cosine_similarity(query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
    """Return cosine scores; normalized dot products equal cosine similarity."""
    query = normalize_vectors(query_vector)
    documents = normalize_vectors(document_vectors)
    if query.shape[1] != documents.shape[1]:
        raise ValueError("Query and document embeddings have different dimensions.")
    return (documents @ query[0]).astype(np.float32)


def retrieve(
    query_vector: np.ndarray,
    document_vectors: np.ndarray,
    chunks: list[Chunk],
    top_k: int = 3,
    min_similarity: float = 0.2,
    redundancy_threshold: float = 0.985,
) -> list[RetrievedChunk]:
    if top_k < 1:
        raise ValueError("Top-K must be at least 1.")
    if len(chunks) != len(document_vectors):
        raise ValueError("Each chunk must have exactly one document embedding.")
    if not chunks:
        return []

    scores = cosine_similarity(query_vector, document_vectors)
    document_vectors = normalize_vectors(document_vectors)
    ranked = np.argsort(-scores, kind="stable")
    selected_indices = []
    seen_text = set()
    for index in ranked:
        score = float(scores[index])
        if score < min_similarity:
            break
        normalized_text = " ".join(chunks[index].text.casefold().split())
        if normalized_text in seen_text:
            continue
        if any(
            float(document_vectors[index] @ document_vectors[chosen]) >= redundancy_threshold
            for chosen in selected_indices
        ):
            continue
        selected_indices.append(int(index))
        seen_text.add(normalized_text)
        if len(selected_indices) == top_k:
            break

    return [RetrievedChunk(chunks[index], float(scores[index])) for index in selected_indices]