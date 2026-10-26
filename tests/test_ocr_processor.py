import pymupdf
import pytest

from src.data_processing import ocr_processor


def _make_pdf(path, text=None):
    document = pymupdf.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_native_pdf_text_does_not_require_tesseract(tmp_path, monkeypatch):
    pdf_path = tmp_path / "contract.pdf"
    output_path = tmp_path / "ocr.txt"
    _make_pdf(pdf_path, "1.1 Term. This agreement begins on the Effective Date.")
    monkeypatch.setattr(
        ocr_processor.pytesseract,
        "image_to_string",
        lambda _image: pytest.fail("OCR should not run for text-bearing pages"),
    )

    text = ocr_processor.process_pdf(pdf_path, output_path)

    assert "This agreement begins" in text
    assert output_path.read_text(encoding="utf-8") == text


def test_scanned_pdf_page_falls_back_to_ocr(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scan.pdf"
    output_path = tmp_path / "ocr.txt"
    _make_pdf(pdf_path)
    monkeypatch.setattr(
        ocr_processor.pytesseract,
        "image_to_string",
        lambda _image: "1.1 Notice. Provide written notice within ten days.",
    )

    text = ocr_processor.process_pdf(pdf_path, output_path)

    assert "Provide written notice" in text


def test_unreadable_pdf_has_clear_error(tmp_path):
    pdf_path = tmp_path / "bad.pdf"
    pdf_path.write_bytes(b"not a pdf")

    with pytest.raises(ValueError, match="not a readable PDF"):
        ocr_processor.process_pdf(pdf_path, tmp_path / "ocr.txt")
