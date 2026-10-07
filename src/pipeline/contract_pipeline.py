from pathlib import Path

from src.data_processing.clause_processor import (
    load_ocr_text,
    extract_clauses,
)

from src.data_processing.contract_processor import (
    process_contract,
)

from src.risk.contract_report import (
    load_clause_results,
    build_contract_report,
    build_contract_risk_summary,
    save_contract_report,
)

from src.risk.review_report import (
    build_review_report,
    save_review_report,
)

from src.retrieval.vector_store import ClauseVectorStore

CLAUSE_OUTPUT_PATH = Path("data/processed/contract_clause_intelligence.json")
REPORT_OUTPUT_PATH = Path("data/processed/contract_intelligence_report.json")
REVIEW_OUTPUT_PATH = Path("data/processed/contract_review_report.json")
VECTOR_INDEX_PATH = Path("data/vector_store/contract_clauses.index")
VECTOR_METADATA_PATH = Path("data/vector_store/contract_clauses.json")


def run_pipeline(
    ocr_input_path=Path("data/processed/ocr_sample.txt"),
    clause_output_path=CLAUSE_OUTPUT_PATH,
    report_output_path=REPORT_OUTPUT_PATH,
    review_output_path=REVIEW_OUTPUT_PATH,
    vector_index_path=VECTOR_INDEX_PATH,
    vector_metadata_path=VECTOR_METADATA_PATH,
):
    """Run the complete Contract Intelligence pipeline.

    Pipeline:

        OCR text
            ↓
        Clause extraction
            ↓
        Clause-level Contract Intelligence
            ↓
        Contract-level aggregation
            ↓
        Human-review report
            ↓
        Embeddings + FAISS Vector Store
    """

    print("AI Contract Intelligence - end-to-end pipeline")

    # STEP 1 — Load OCR text
    print("\n[1/4] Loading OCR text...")

    ocr_text = load_ocr_text(ocr_input_path)
    clauses = extract_clauses(ocr_text)

    print(f"OCR text loaded. {len(clauses)} clauses detected.")

    if not clauses:
        raise ValueError("No clauses were extracted from the OCR text.")

    # STEP 2 — Run clause-level Contract Intelligence
    print("\n[2/4] Running clause-level intelligence...")

    process_contract(
        ocr_input_path=ocr_input_path,
        output_path=clause_output_path,
    )

    if not clause_output_path.exists():
        raise FileNotFoundError("Clause intelligence output was not generated.")
    print("\n Clause-level intelligence completed.")

    # STEP 3 — Build contract-level reports
    print("\n[3/4] Building contract-level reports...")

    clause_results = load_clause_results(clause_output_path)
    report = build_contract_report(clause_results)

    print("\nContract-level report generated.")

    review_report = build_review_report(clause_results)
    save_review_report(review_report, review_output_path)

    report["risk_summary"] = build_contract_risk_summary(report, review_report)
    save_contract_report(report, report_output_path)

    print("Human-review report generated.")

    # STEP 4 — Build semantic vector store
    print("\n[4/4] Building semantic vector store...")

    vector_store = ClauseVectorStore(
        index_path=vector_index_path,
        metadata_path=vector_metadata_path,
    )

    vector_store.build(clauses)

    print("\nSemantic vector store generated successfully.")
    print("\nEnd-to-end pipeline completed successfully.")
    print(f"\nClauses detected: {len(clauses)}")
    print(
        "Clauses processed: " f"{report['contract_summary']['successfully_processed']}"
    )
    print("Processing errors: " f"{report['contract_summary']['processing_errors']}")
    print("Low-confidence clauses: " f"{len(report['low_confidence_clauses'])}")

    print(f"\nClause-level output:\n{clause_output_path}")
    print(f"\nContract-level output:\n{report_output_path}")
    print(f"\nHuman-review output:\n{review_output_path}")
    print(f"\nVector index:\n{VECTOR_INDEX_PATH}")
    print(f"\nVector metadata:\n{VECTOR_METADATA_PATH}")

    return {
        "contract_report": report,
        "review_report": review_report,
        "clauses": clauses,
    }


if __name__ == "__main__":
    run_pipeline()
