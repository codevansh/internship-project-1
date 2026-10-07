import json
from pathlib import Path
from typing import Dict, List

INPUT_PATH = Path("data/processed/contract_clause_intelligence.json")
OUTPUT_PATH = Path("data/processed/contract_review_report.json")

CONFIDENCE_THRESHOLD = 0.70


def load_clause_results(
    input_path: Path = INPUT_PATH,
) -> List[Dict]:
    """Load clause-level Contract Intelligence results."""

    if not input_path.exists():
        raise FileNotFoundError(f"Clause intelligence file not found: {input_path}")
    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as file:
        results = json.load(file)

    if not isinstance(results, list):
        raise ValueError("Clause intelligence output must contain a list.")

    return results


def build_review_items(
    clauses: List[Dict],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> List[Dict]:
    """
    Build human-review items for clauses whose
    classification confidence is below the threshold.
    """

    review_items = []

    for clause in clauses:
        confidence = clause.get("classification_confidence")
        if confidence is None:
            continue
        if confidence >= threshold:
            continue
        review_item = {
            "article": clause.get("article"),
            "article_title": clause.get("article_title"),
            "section": clause.get("section"),
            "section_title": clause.get("section_title"),
            "schedule": clause.get("schedule"),
            "clause_number": clause.get("clause_number"),
            "title": clause.get("title"),
            "predicted_clause_type": clause.get("clause_type"),
            "classification_confidence": confidence,
            "classification_validation": clause.get("classification_validation", {}),
            "review_recommendation": "Review classifier output and clause evidence.",
            "top_predictions": clause.get(
                "top_predictions",
                [],
            ),
            "clause_text": clause.get(
                "clause_text",
                "",
            ),
            "entities": clause.get(
                "entities",
                [],
            ),
            "contractual_indicators": clause.get(
                "contractual_indicators",
                {},
            ),
        }
        review_items.append(review_item)

    # Lowest confidence first.
    review_items.sort(key=lambda item: item["classification_confidence"])
    return review_items


def build_review_report(
    clauses: List[Dict],
    threshold: float = CONFIDENCE_THRESHOLD,
) -> Dict:
    """Build the contract human-review report."""

    review_items = build_review_items(clauses, threshold)
    successfully_processed = [clause for clause in clauses if "error" not in clause]

    return {
        "review_summary": {
            "total_clauses": len(clauses),
            "successfully_processed": len(successfully_processed),
            "confidence_threshold": threshold,
            "clauses_requiring_review": len(review_items),
        },
        "review_items": review_items,
    }


def save_review_report(
    report: Dict,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Save the human-review report as JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )


def main():
    print("CONTRACT HUMAN REVIEW REPORT")
    print("\n[1/3] Loading clause intelligence...")

    clauses = load_clause_results()
    print(f"Loaded {len(clauses)} clause results.")

    print("\n[2/3] Identifying clauses for review...")
    report = build_review_report(clauses)

    print("Review report generated.")
    print("\n[3/3] Saving review report...")
    save_review_report(report)
    print(f"\nOutput saved to:\n" f"{OUTPUT_PATH}")

    print("Total clauses:", report["review_summary"]["total_clauses"])

    print("Confidence threshold:", report["review_summary"]["confidence_threshold"])

    print(
        "Clauses requiring review:",
        report["review_summary"]["clauses_requiring_review"],
    )

    print("\n Review queue:")

    for item in report["review_items"]:
        print(
            f"{item['clause_number']} - "
            f"{item['title']} | "
            f"{item['predicted_clause_type']} | "
            f"{item['classification_confidence']:.4f}"
        )

    print("\n Human-review report completed.")


if __name__ == "__main__":
    main()
