import pytest

from core.chunker import create_chunks
from core.document_loader import ExtractedPage
from core.text_cleaner import clean_text


def test_clean_text_preserves_paragraphs_and_joins_wrapped_lines():
    assert clean_text("  First  line\ncontinued\n\nSecond\x00 paragraph ") == (
        "First line continued\n\nSecond paragraph"
    )


def test_chunks_overlap_and_keep_page_metadata():
    pages = [ExtractedPage("one two three four five six seven", 4)]
    chunks = create_chunks(pages, "course.pdf", chunk_size=4, overlap=2)

    assert [chunk.text for chunk in chunks] == ["one two three four", "three four five six", "five six seven"]
    assert all(chunk.source == "course.pdf" and chunk.page_number == 4 for chunk in chunks)
    assert chunks[0].chunk_id.startswith("course.pdf-p4-c")


def test_chunk_configuration_is_validated():
    with pytest.raises(ValueError):
        create_chunks([ExtractedPage("some text", 1)], "x.pdf", chunk_size=3, overlap=3)