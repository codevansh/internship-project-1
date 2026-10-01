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
    save_contract_report,
)

CLAUSE_OUTPUT_PATH = Path("data/processed/contract_clause_intelligence.json")
REPORT_OUTPUT_PATH = Path("data/processed/contract_intelligence_report.json")


def run_pipeline():
    """Run the complete Contract Intelligence pipeline.

    Pipeline:

        OCR text
            ↓
        Clause extraction
            ↓
        Clause-level Contract Intelligence
            ↓
        Contract-level aggregation
    """

    print("Ai contract intelligence - end-to-end pipeline")

    # STEP 1 — Load OCR text
    print("\n [1/3] Loading OCR text...")

    ocr_text = load_ocr_text()
    clauses = extract_clauses(ocr_text)
    print(f"OCR text loaded." f" {len(clauses)} clauses detected.")

    if not clauses:
        raise ValueError("No clauses were extracted from the OCR text.")

    # STEP 2 — Run clause-level Contract Intelligence
    print("\n [2/3] Running clause-level intelligence...")

    process_contract()

    if not CLAUSE_OUTPUT_PATH.exists():
        raise FileNotFoundError("Clause intelligence output was not generated.")
    print("\n Clause-level intelligence completed.")

    # STEP 3 — Build contract-level report
    print("\n [3/3] Building contract-level report...")

    clause_results = load_clause_results(CLAUSE_OUTPUT_PATH)
    report = build_contract_report(clause_results)
    save_contract_report(report, REPORT_OUTPUT_PATH)

    print("\nContract-level report generated.")

    # FINAL SUMMARY
    print("END-TO-END PIPELINE COMPLETED")

    print(f"\n Clauses detected: " f"{len(clauses)}")
    print(f"Clauses processed: " f"{report['contract_summary']['successfully_processed']}")
    print(f"Processing errors: " f"{report['contract_summary']['processing_errors']}")
    print(f"Low-confidence clauses: " f"{len(report['low_confidence_clauses'])}")
    print(f"\n Clause-level output:\n" f"{CLAUSE_OUTPUT_PATH}")
    print(f"\n Contract-level output:\n" f"{REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    run_pipeline()
