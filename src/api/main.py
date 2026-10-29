from pathlib import Path
import os
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.data_processing.ocr_processor import process_pdf
from src.pipeline.contract_pipeline import run_pipeline

MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024

app = FastAPI(
    title="AI Contract Intelligence API",
    description="API for contract analysis and intelligence extraction.",
    version="1.0.0",
)

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
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
            pdf_path = temp_dir / "upload.pdf"
            ocr_path = temp_dir / "ocr.txt"

            pdf_bytes = await file.read(MAX_UPLOAD_SIZE_BYTES + 1)
            if not pdf_bytes:
                raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
            if len(pdf_bytes) > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(status_code=413, detail="The PDF must be 25 MB or smaller.")
            pdf_path.write_bytes(pdf_bytes)
            del pdf_bytes

            process_pdf(pdf_path, ocr_path)

            # Keep per-upload artifacts isolated: concurrent users must not
            # overwrite each other's reports or the checked-in sample outputs.
            result = run_pipeline(
                ocr_input_path=ocr_path,
                clause_output_path=temp_dir / "clauses.json",
                report_output_path=temp_dir / "contract_report.json",
                review_output_path=temp_dir / "review_report.json",
                vector_index_path=temp_dir / "clauses.index",
                vector_metadata_path=temp_dir / "clauses.json.metadata",
            )

            contract_report = result["contract_report"]
            summary = contract_report.get("contract_summary", {})
            if "contract_summary" in contract_report and summary.get("successfully_processed", 0) == 0:
                raise RuntimeError(
                    "No clauses could be analyzed successfully. Check the model files and processing dependencies."
                )

            return {
                "filename": file.filename,
                "contract_report": contract_report,
                "review_report": result["review_report"],
                "risk_scoring": contract_report.get("risk_scoring", {}),
                "risk_summary": contract_report.get("risk_summary", {}),
            }
    except HTTPException:
        raise
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Contract analysis failed: {error}",
        ) from error
