import numpy as np
import pytest

from core import embeddings


def test_embedding_generation_normalizes_local_vectors(monkeypatch):
    class FakeModel:
        def encode(self, texts, **kwargs):
            assert texts == ["first", "second"]
            assert kwargs["show_progress_bar"] is False
            return np.array([[3.0, 4.0], [0.0, 2.0]])

    monkeypatch.setattr(embeddings, "_load_model", lambda: FakeModel())
    vectors = embeddings.embed_texts(["first", "second"])

    assert np.allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])
    assert np.allclose(vectors[0], [0.6, 0.8])


def test_empty_embedding_input_does_not_load_model(monkeypatch):
    monkeypatch.setattr(embeddings, "_load_model", lambda: pytest.fail("model should not load"))
    assert embeddings.embed_texts([]).shape == (0, 0)