from pathlib import Path
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.data_processing.ocr_processor import process_pdf
from src.pipeline.contract_pipeline import run_pipeline

app = FastAPI(
    title="AI Contract Intelligence API",
    description="API for contract analysis and intelligence extraction.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "contract-intelligence-api",
    }


@app.post("/analyze")
async def analyze_contract(file: UploadFile = File(...)):
    """Analyze an uploaded contract PDF."""

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            pdf_path = temp_dir / file.filename
            ocr_path = temp_dir / "ocr.txt"

            # Save uploaded PDF
            pdf_bytes = await file.read()
            pdf_path.write_bytes(pdf_bytes)

            # Run OCR
            process_pdf(pdf_path, ocr_path)

            # Run Contract Intelligence pipeline
            result = run_pipeline(ocr_input_path=ocr_path)

            return {
                "filename": file.filename,
                "contract_report": result["contract_report"],
                "review_report": result["review_report"],
                "risk_scoring": result["contract_report"].get("risk_scoring", {}),
                "risk_summary": result["contract_report"].get("risk_summary", {}),
            }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )
