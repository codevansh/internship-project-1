from pathlib import Path
from pdf2image import convert_from_path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PDF_PATH = Path("data/raw/sample_contract.pdf")
OCR_OUTPUT_PATH = Path("data/processed/ocr_sample.txt")
POPPLER_PATH=r"C:\Users\Lenovo\Downloads\Release-26.07.0-0\poppler-26.07.0\Library\bin"


def pdf_to_images(pdf_path):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")
    
    images = convert_from_path(pdf_path,poppler_path=POPPLER_PATH)
    print(f"PDF Found: {pdf_path}")
    print(f"Pages converted: {len(images)}")
    
    return images


def extract_text_from_images(images):
    extracted_text =[]
    
    for page_number,image in enumerate(images,start=1):
        text = pytesseract.image_to_string(image)
        extracted_text.append(f"\n Page {page_number} \n {text}")
        
        print(f"OCR completed for page {page_number}")
    return '\n'.join(extracted_text)


def save_ocr_text(text,output_path):
    output_path.parent.mkdir(parents=True,exist_ok=True)
    
    with open(output_path,'w',encoding='utf-8') as file:
        file.write(text)
        
    print(f"OCR text saved to:")
    print(output_path)
    

def main():
    images=pdf_to_images(PDF_PATH)
    
    extracted_text = extract_text_from_images(images)
    
    save_ocr_text(extracted_text,OCR_OUTPUT_PATH)
    print("\n OCR processing completed successfully")
    

if __name__ == "__main__":
    main()