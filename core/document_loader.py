from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from core.text_cleaner import clean_text


@dataclass(frozen=True)
class ExtractedPage:
    text: str
    page_number: int | None


def extract_document(content: bytes, filename: str) -> list[ExtractedPage]:
    """Extract text from a PDF or DOCX byte stream without writing it to disk."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz

            with fitz.open(stream=content, filetype="pdf") as pdf:
                pages = [
                    ExtractedPage(clean_text(page.get_text("text")), index + 1)
                    for index, page in enumerate(pdf)
                ]
        except Exception as exc:
            raise ValueError(f"Could not read this PDF: {exc}") from exc
        if not pages:
            raise ValueError("This PDF has no pages to read.")
        if not any(page.text for page in pages):
            raise ValueError(
                "No selectable text was found. This may be an image-only PDF; "
                "OCR is not included in this lightweight workshop app."
            )
        return pages

    if suffix == ".docx":
        try:
            from docx import Document

            document = Document(BytesIO(content))
            sections = [paragraph.text for paragraph in document.paragraphs]
            for table in document.tables:
                for row in table.rows:
                    sections.append(" | ".join(cell.text for cell in row.cells))
            text = clean_text("\n\n".join(sections))
        except Exception as exc:
            raise ValueError(f"Could not read this DOCX: {exc}") from exc
        if not text:
            raise ValueError("No readable text was found in this DOCX.")
        return [ExtractedPage(text, None)]

    raise ValueError("Unsupported file type. Please upload a PDF or DOCX.")