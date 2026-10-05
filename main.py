from pathlib import Path
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.data_processing.ocr_processor import process_pdf
from src.pipeline.contract_pipeline import run_pipeline

app = FastAPI(
    title="AI Contract Intelligence and Risk Scoring API",
    description="API for contract OCR, intelligence, and risk analysis.",
)


@app.get("/")
def root():
    return {"message": "Project running successfully", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze-contract")
async def analyze_contract(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            pdf_path = temp_dir / file.filename
            ocr_path = temp_dir / "ocr_output.txt"

            file_content = await file.read()
            pdf_path.write_bytes(file_content)

            process_pdf(
                pdf_path=pdf_path,
                output_path=ocr_path,
            )

            result = run_pipeline(
                ocr_input_path=ocr_path,
                clause_output_path=temp_dir / "clause_intelligence.json",
                report_output_path=temp_dir / "contract_report.json",
                review_output_path=temp_dir / "review_report.json",
                vector_index_path=temp_dir / "contract_clauses.index",
                vector_metadata_path=temp_dir / "contract_clauses.json",
            )

            return {
                "filename": file.filename,
                "status": "success",
                "contract_report": result["contract_report"],
                "review_report": result["review_report"],
                "clauses_detected": len(result["clauses"]),
            }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Contract processing failed: {str(exc)}",
        )
