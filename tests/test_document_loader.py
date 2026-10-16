import pytest
from pypdf import PdfWriter

from rag.document_loader import load_document

def test_load_pdf(tmp_path):
    file_path = tmp_path / "policy.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)

    # A blank PDF contains no extractable text.
    with file_path.open("wb") as pdf_file:
        writer.write(pdf_file)

    with pytest.raises(ValueError, match="No extractable text"):
        load_document(str(file_path))
