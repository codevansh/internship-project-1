from pathlib import Path
from io import BytesIO
from PIL import Image

import pymupdf
import pytesseract

# Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Default paths for standalone testing
PDF_PATH = Path("data/raw/sample_contract.pdf")
OCR_OUTPUT_PATH = Path("data/processed/ocr_sample.txt")


def pdf_to_images(pdf_path):
    """Convert every PDF page into a PIL image using PyMuPDF."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")
    document = pymupdf.open(pdf_path)
    images = []

    for page in document:
        # Render page at 2x resolution for better OCR quality
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(3, 3))
        # Convert rendered page into a PIL image
        image = Image.open(BytesIO(pixmap.tobytes("png")))
        images.append(image)
    document.close()

    print(f"PDF Found: {pdf_path}")
    print(f"Converting Pages: {len(images)}")

    return images


def extract_text_from_images(images):
    """Extract text from all page images using Tesseract OCR."""

    extracted_text = []

    for page_number, image in enumerate(images, start=1):
        text = pytesseract.image_to_string(image)
        extracted_text.append(f"\nPage {page_number}\n{text}")
        print(f"OCR completed for page {page_number}")
    return "\n".join(extracted_text)


def save_ocr_text(text, output_path):
    """Save OCR text to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(text)
    print("OCR text saved to:")
    print(output_path)


def process_pdf(pdf_path, output_path):
    """Run the complete PDF → OCR → text pipeline."""

    images = pdf_to_images(pdf_path)
    extracted_text = extract_text_from_images(images)
    save_ocr_text(extracted_text, output_path)
    print("\n OCR processing completed successfully")
    return extracted_text


def main():
    """Run OCR using the default sample contract."""
    process_pdf(PDF_PATH, OCR_OUTPUT_PATH)


if __name__ == "__main__":
    main()
