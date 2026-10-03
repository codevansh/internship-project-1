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

from src.risk.review_report import (
    build_review_report,
    save_review_report,
)

CLAUSE_OUTPUT_PATH = Path("data/processed/contract_clause_intelligence.json")
REPORT_OUTPUT_PATH = Path("data/processed/contract_intelligence_report.json")
REVIEW_OUTPUT_PATH = Path("data/processed/contract_review_report.json")


def run_pipeline(
    ocr_input_path=Path("data/processed/ocr_sample.txt"),
    clause_output_path=CLAUSE_OUTPUT_PATH,
    report_output_path=REPORT_OUTPUT_PATH,
    review_output_path=REVIEW_OUTPUT_PATH,
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
    """

    print("AI Contract Intelligence - end-to-end pipeline")

    # STEP 1 — Load OCR text
    print("\n[1/3] Loading OCR text...")

    ocr_text = load_ocr_text(ocr_input_path)
    clauses = extract_clauses(ocr_text)

    print(f"OCR text loaded. {len(clauses)} clauses detected.")

    if not clauses:
        raise ValueError("No clauses were extracted from the OCR text.")

    # STEP 2 — Run clause-level Contract Intelligence
    print("\n[2/3] Running clause-level intelligence...")

    process_contract(
        ocr_input_path=ocr_input_path,
        output_path=clause_output_path,
    )

    if not clause_output_path.exists():
        raise FileNotFoundError("Clause intelligence output was not generated.")
    print("\nClause-level intelligence completed.")

    # STEP 3 — Build contract-level report
    print("\n[3/3] Building contract-level report...")
    clause_results = load_clause_results(clause_output_path)
    report = build_contract_report(clause_results)
    save_contract_report(report, report_output_path)
    print("\n Contract-level report generated")

    # Human-review report
    review_report = build_review_report(clause_results)
    save_review_report(review_report, review_output_path)
    print("Human-review report generated.")

    # FINAL SUMMARY
    print("\n End-to-end pipeline completed")

    print(f"\n Clauses detected: {len(clauses)}")
    print(
        "Clauses processed: " f"{report['contract_summary']['successfully_processed']}"
    )
    print("Processing errors: " f"{report['contract_summary']['processing_errors']}")
    print("Low-confidence clauses: " f"{len(report['low_confidence_clauses'])}")

    print(f"\n Clause-level output:\n{clause_output_path}")
    print(f"\n Contract-level output:\n{report_output_path}")
    print(f"\n Human-review output:\n{review_output_path}")

    return {
        "contract_report": report,
        "review_report": review_report,
        "clauses": clauses,
    }


if __name__ == "__main__":
    run_pipeline()
