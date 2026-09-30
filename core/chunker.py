from dataclasses import dataclass
from pathlib import Path

from core.document_loader import ExtractedPage
from core.text_cleaner import clean_text


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    page_number: int | None
    text: str


def create_chunks(
    pages: list[ExtractedPage],
    source: str,
    chunk_size: int = 180,
    overlap: int = 35,
) -> list[Chunk]:
    """Split pages into overlapping word windows, retaining page provenance."""
    if chunk_size < 1:
        raise ValueError("Chunk size must be at least 1 word.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be zero or more and smaller than chunk size.")

    filename = Path(source).name
    chunks = []
    stride = chunk_size - overlap
    for page in pages:
        words = clean_text(page.text).split()
        for start in range(0, len(words), stride):
            text = " ".join(words[start : start + chunk_size])
            if not text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{filename}-p{page.page_number or 'x'}-c{len(chunks) + 1}",
                    source=filename,
                    page_number=page.page_number,
                    text=text,
                )
            )
            if start + chunk_size >= len(words):
                break
    return chunks