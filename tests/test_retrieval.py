import numpy as np

from core.chunker import Chunk
from core.retriever import cosine_similarity, retrieve


def test_cosine_similarity_and_top_k_order():
    chunks = [Chunk("a", "doc.pdf", 1, "agriculture"), Chunk("b", "doc.pdf", 2, "rainfall")]
    vectors = np.array([[1, 0], [0, 1]], dtype=np.float32)

    scores = cosine_similarity(np.array([1, 0]), vectors)
    found = retrieve(np.array([1, 0]), vectors, chunks, top_k=2, min_similarity=0.1)

    assert np.allclose(scores, [1, 0])
    assert [item.chunk.chunk_id for item in found] == ["a"]


def test_threshold_can_return_no_sources():
    chunks = [Chunk("a", "doc.pdf", 1, "content")]
    found = retrieve(np.array([0, 1]), np.array([[1, 0]]), chunks, min_similarity=0.2)
    assert found == []


def test_duplicate_text_is_not_retrieved_twice():
    chunks = [Chunk("a", "doc.pdf", 1, "same text"), Chunk("b", "doc.pdf", 1, " same   text ")]
    found = retrieve(np.array([1, 0]), np.array([[1, 0], [1, 0]]), chunks, top_k=2)
    assert [item.chunk.chunk_id for item in found] == ["a"]