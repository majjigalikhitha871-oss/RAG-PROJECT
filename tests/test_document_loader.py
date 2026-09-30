from io import BytesIO

import fitz
import pytest
from docx import Document

from core.document_loader import extract_document


def test_extracts_pdf_text_and_page_number():
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Eligibility requires two years of study.")
    content = pdf.tobytes()
    pdf.close()

    result = extract_document(content, "rules.pdf")

    assert result[0].page_number == 1
    assert "two years of study" in result[0].text


def test_extracts_docx_paragraphs_and_table_text():
    document = Document()
    document.add_paragraph("Admission requirements")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Minimum age"
    table.cell(0, 1).text = "18"
    stream = BytesIO()
    document.save(stream)

    result = extract_document(stream.getvalue(), "rules.docx")

    assert result[0].page_number is None
    assert "Admission requirements" in result[0].text
    assert "Minimum age | 18" in result[0].text


def test_empty_pdf_has_friendly_error():
    pdf = fitz.open()
    pdf.new_page()
    content = pdf.tobytes()
    pdf.close()

    with pytest.raises(ValueError, match="No selectable text"):
        extract_document(content, "scan.pdf")


def test_rejects_unsupported_file():
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_document(b"text", "notes.txt")