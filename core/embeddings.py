from functools import lru_cache

import numpy as np


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("Embeddings must be a one- or two-dimensional array.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms != 0)


@lru_cache(maxsize=1)
def _load_model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "The local embedding model is not ready. Install the dependencies, "
            "then run the explicit download command in README.md while online."
        ) from exc


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model = _load_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return normalize_vectors(vectors)