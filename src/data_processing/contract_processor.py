from pathlib import Path

from src.data_processing.clause_processor import (
    load_ocr_text,
    extract_clauses,
)

from src.risk.contract_intelligence import (
    load_contract_intelligence_models,
    analyze_clause,
    save_results,
)

OUTPUT_PATH = Path("data/processed/contract_clause_intelligence.json")


def process_contract(
    ocr_input_path=Path("data/processed/ocr_sample.txt"), output_path=OUTPUT_PATH
):
    """Process an OCR-extracted contract.

    Pipeline:

        OCR text
            ↓
        Clause extraction
            ↓
        Legal-RoBERTa classification
            ↓
        spaCy NER
            ↓
        Contractual indicators
            ↓
        Clause-level intelligence
    """

    # 1. Load OCR text
    print("\n[1/4] Loading OCR text...")
    ocr_text = load_ocr_text(ocr_input_path)
    print("OCR text loaded successfully.")

    # 2. Extract clauses
    print("\n[2/4] Extracting clauses...")
    clauses = extract_clauses(ocr_text)
    print(f"Clauses extracted: {len(clauses)}")

    if not clauses:
        raise ValueError("No clauses were extracted from the OCR text.")

    # 3. Load Contract Intelligence models
    print("\n[3/4] Loading Contract Intelligence models...")

    tokenizer, model, id_to_label, nlp = load_contract_intelligence_models()
    print("Models loaded successfully.")

    # 4. Analyze every clause
    print("\n[4/4] Running Contract Intelligence...")
    results = []

    for index, clause in enumerate(clauses, start=1):

        print(
            f"\nProcessing clause "
            f"{index}/{len(clauses)}: "
            f"{clause['clause_number']} - "
            f"{clause['title']}"
        )

        try:
            intelligence = analyze_clause(
                clause["text"],
                tokenizer,
                model,
                id_to_label,
                nlp,
            )

            # Preserve document structure.
            intelligence["article"] = clause["article"]
            intelligence["clause_number"] = clause["clause_number"]
            intelligence["title"] = clause["title"]

            results.append(intelligence)

            print(f"Prediction: " f"{intelligence['clause_type']}")
            print(f"Confidence: " f"{intelligence['classification_confidence']:.4f}")

        except Exception as error:
            print(f"  ERROR processing clause " f"{clause['clause_number']}: {error}")

            results.append(
                {
                    "article": clause["article"],
                    "clause_number": clause["clause_number"],
                    "title": clause["title"],
                    "error": str(error),
                }
            )

    # Save results
    save_results(results, output_path=output_path)

    print("Contract processing completed")
    print(f"\nTotal clauses: {len(clauses)}")
    print(f"Results generated: {len(results)}")
    print(f"\nOutput saved to:\n" f"{output_path}")
    return results


if __name__ == "__main__":
    process_contract()
