import json
from pathlib import Path
from typing import Dict, List

INPUT_PATH = Path("data/processed/contract_clause_intelligence.json")

OUTPUT_PATH = Path("data/processed/contract_intelligence_report.json")


def load_clause_results(
    input_path: Path = INPUT_PATH,
) -> List[Dict]:
    """Load clause-level Contract Intelligence results."""

    if not input_path.exists():
        raise FileNotFoundError(f"Clause intelligence file not found: {input_path}")
    with open(input_path, "r", encoding="utf-8") as file:
        results = json.load(file)
    if not isinstance(results, list):
        raise ValueError("Clause intelligence output must contain a list.")
    return results


def count_clause_types(
    clauses: List[Dict],
) -> Dict[str, int]:
    """Count how many times each clause type was predicted."""

    counts = {}

    for clause in clauses:
        clause_type = clause.get("clause_type")
        if not clause_type:
            continue
        counts[clause_type] = counts.get(clause_type, 0) + 1

    return dict(
        sorted(
            counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    )


def find_low_confidence_clauses(
    clauses: List[Dict],
    threshold: float = 0.70,
) -> List[Dict]:
    """
    Find clauses whose classification confidence
    is below the review threshold.
    """

    low_confidence = []

    for clause in clauses:
        confidence = clause.get("classification_confidence")

        if confidence is None:
            continue
        if confidence < threshold:
            low_confidence.append(
                {
                    "article": clause.get("article"),
                    "clause_number": clause.get("clause_number"),
                    "title": clause.get("title"),
                    "clause_type": clause.get("clause_type"),
                    "classification_confidence": confidence,
                }
            )

    return low_confidence


def collect_entities(
    clauses: List[Dict],
) -> List[Dict]:
    """Collect entities detected across the contract."""

    entities = []

    for clause in clauses:
        for entity in clause.get("entities", []):
            entities.append(
                {
                    "article": clause.get("article"),
                    "clause_number": clause.get("clause_number"),
                    "entity": entity,
                }
            )
    return entities


def summarize_entities(
    clauses: List[Dict],
) -> Dict[str, Dict]:
    """
    Create a contract-level summary of unique entities.

    Each entity is grouped by its text and label,
    while repeated occurrences are counted.
    """

    summary = {}

    for clause in clauses:
        for entity in clause.get("entities", []):
            entity_text = entity.get("text")
            label = entity.get("label")

            if not entity_text or not label:
                continue
            key = f"{entity_text}|{label}"
            if key not in summary:
                summary[key] = {
                    "text": entity_text,
                    "label": label,
                    "description": entity.get(
                        "description",
                        "",
                    ),
                    "occurrences": 0,
                }
            summary[key]["occurrences"] += 1
    return dict(
        sorted(
            summary.items(),
            key=lambda item: item[1]["occurrences"],
            reverse=True,
        )
    )


def collect_contractual_indicators(
    clauses: List[Dict],
) -> List[Dict]:
    """Collect contractual indicators across clauses."""

    indicators = []

    for clause in clauses:
        clause_indicators = clause.get(
            "contractual_indicators",
            {},
        )
        if not clause_indicators:
            continue
        indicators.append(
            {
                "article": clause.get("article"),
                "clause_number": clause.get("clause_number"),
                "title": clause.get("title"),
                "indicators": clause_indicators,
            }
        )
    return indicators


def build_contract_report(
    clauses: List[Dict],
) -> Dict:
    """Build the contract-level intelligence report."""

    successful_clauses = [clause for clause in clauses if "error" not in clause]
    report = {
        "contract_summary": {
            "total_clauses": len(clauses),
            "successfully_processed": len(successful_clauses),
            "processing_errors": (len(clauses) - len(successful_clauses)),
        },
        "clause_type_summary": count_clause_types(successful_clauses),
        "low_confidence_clauses": (find_low_confidence_clauses(successful_clauses)),
        "entities": collect_entities(successful_clauses),
        "entity_summary": summarize_entities(successful_clauses),
        "contractual_indicators": (collect_contractual_indicators(successful_clauses)),
        "clauses": successful_clauses,
        "risk_assessments": collect_risk_assessments(successful_clauses)
    }
    return report


def collect_risk_assessments(
    clauses: List[Dict],
) -> List[Dict]:
    """Collect clause-level risk assessments across the contract."""

    assessments = []

    for clause in clauses:
        risk_assessment = clause.get(
            "risk_assessment",
            {},
        )

        if not risk_assessment:
            continue

        assessments.append(
            {
                "article": clause.get("article"),
                "clause_number": clause.get("clause_number"),
                "title": clause.get("title"),
                "clause_type": clause.get("clause_type"),
                "classification_confidence": clause.get("classification_confidence"),
                "risk_assessment": risk_assessment,
            }
        )

    return assessments


def save_contract_report(
    report: Dict,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Save the contract-level report as JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)


def main():

    print("CONTRACT-LEVEL INTELLIGENCE REPORT")

    print("\n[1/3] Loading clause intelligence...")

    clauses = load_clause_results()
    print(f"Loaded {len(clauses)} clause results.")

    print("\n[2/3] Building contract-level report...")

    report = build_contract_report(clauses)
    print("Contract report generated.")

    print("\n[3/3] Saving report...")
    save_contract_report(report)
    print(f"\nOutput saved to:\n" f"{OUTPUT_PATH}")

    print("\n Total clauses:", report["contract_summary"]["total_clauses"])
    print(
        "\n Successfully processed:",
        report["contract_summary"]["successfully_processed"],
    )
    print("\n Processing errors:", report["contract_summary"]["processing_errors"])
    print("\n Low-confidence clauses:", len(report["low_confidence_clauses"]))

    print("\nClause type summary:")
    for clause_type, count in report["clause_type_summary"].items():
        print(f"  {clause_type}: {count}")


if __name__ == "__main__":
    main()
