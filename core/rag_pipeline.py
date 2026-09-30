from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.chunker import Chunk
from core.retriever import RetrievedChunk, retrieve
from core.text_cleaner import clean_text


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    sources: list[RetrievedChunk]


def answer_question(
    question: str,
    chunks: list[Chunk],
    document_vectors: np.ndarray,
    embed: Callable[[list[str]], np.ndarray],
    generate: Callable[[str, str], str],
    top_k: int = 3,
    min_similarity: float = 0.2,
) -> AnswerResult:
    cleaned_question = clean_text(question)
    if not cleaned_question:
        raise ValueError("Enter a question before sending it.")
    if not chunks:
        raise ValueError("Upload and index a document before asking a question.")

    query_vector = embed([cleaned_question])
    sources = retrieve(query_vector, document_vectors, chunks, top_k, min_similarity)
    if not sources:
        return AnswerResult(
            "I could not find relevant information in the uploaded documents. "
            "Try a question that is more closely related to their content.",
            [],
        )

    context_parts = []
    for item in sources:
        page = f", page {item.chunk.page_number}" if item.chunk.page_number else ""
        context_parts.append(
            f"[Source: {item.chunk.source}{page}; chunk {item.chunk.chunk_id}]\n{item.chunk.text}"
        )
    return AnswerResult(generate(cleaned_question, "\n\n".join(context_parts)), sources)