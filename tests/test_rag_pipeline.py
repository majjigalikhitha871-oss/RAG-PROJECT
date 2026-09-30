import numpy as np

from core.chunker import Chunk
from core.rag_pipeline import answer_question


def test_retrieved_context_is_sent_for_generation():
    chunks = [Chunk("c1", "guide.pdf", 2, "Applicants must be at least 18 years old.")]
    generation_calls = []
    result = answer_question(
        "What is the minimum age?",
        chunks,
        np.array([[1.0, 0.0]]),
        lambda texts: np.array([[1.0, 0.0]]),
        lambda question, context: generation_calls.append((question, context)) or "At least 18.",
    )

    assert result.answer == "At least 18."
    assert "guide.pdf, page 2" in generation_calls[0][1]
    assert "Applicants must be at least 18" in generation_calls[0][1]


def test_no_relevant_retrieval_does_not_call_generation():
    called = False

    def generate(question, context):
        nonlocal called
        called = True
        return "unexpected"

    result = answer_question(
        "What is the weather?",
        [Chunk("c1", "guide.pdf", 1, "admission requirements")],
        np.array([[1.0, 0.0]]),
        lambda texts: np.array([[0.0, 1.0]]),
        generate,
        min_similarity=0.2,
    )

    assert not called
    assert not result.sources
    assert "could not find relevant" in result.answer