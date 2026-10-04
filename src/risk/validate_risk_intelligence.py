import json
from pathlib import Path

INPUT_PATH = Path("data/processed/contract_clause_intelligence.json")

CONFIDENCE_THRESHOLD = 0.70


def load_results():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Risk Intelligence output not found: {INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        results = json.load(file)
    if not isinstance(results, list):
        raise ValueError("Risk Intelligence output must contain a list.")
    return results


def validate_risk_intelligence(results):
    total = len(results)
    errors = []

    risk_assessments = 0
    review_count = 0
    indicator_count = 0
    entity_count = 0

    for index, clause in enumerate(results, start=1):
        clause_number = clause.get(
            "clause_number",
            f"unknown-{index}",
        )

        # Check required fields
        required_fields = [
            "clause_type",
            "classification_confidence",
            "entities",
            "contractual_indicators",
            "risk_assessment",
            "top_predictions",
        ]

        for field in required_fields:
            if field not in clause:
                errors.append(f"{clause_number}: missing field '{field}'")

        risk_assessment = clause.get("risk_assessment")

        if not isinstance(risk_assessment, dict):
            errors.append(f"{clause_number}: invalid risk_assessment")
            continue

        risk_assessments += 1

        # Check risk assessment fields
        risk_fields = [
            "level",
            "reasons",
            "evidence",
            "requires_human_review",
        ]

        for field in risk_fields:
            if field not in risk_assessment:
                errors.append(f"{clause_number}: missing risk field '{field}'")

        confidence = clause.get("classification_confidence")

        if confidence is None:
            continue

        indicators = clause.get("contractual_indicators",{})
        entities = clause.get("entities",[])

        if indicators:
            indicator_count += 1
        if entities:
            entity_count += 1

        requires_review = risk_assessment.get("requires_human_review")
        expected_review = confidence < CONFIDENCE_THRESHOLD

        if requires_review != expected_review:
            errors.append(
                f"{clause_number}: review flag mismatch "
                f"(confidence={confidence}, "
                f"requires_human_review={requires_review})"
            )

        if requires_review:
            review_count += 1

    return {
        "total": total,
        "risk_assessments": risk_assessments,
        "review_count": review_count,
        "indicator_count": indicator_count,
        "entity_count": entity_count,
        "errors": errors,
    }


def main():

    results = load_results()
    print(f"Loaded clauses: {len(results)}")

    validation = validate_risk_intelligence(results)
    print(f"\nClauses analyzed: " f"{validation['total']}")

    print(f"Risk assessments generated: " f"{validation['risk_assessments']}")
    print(f"Clauses requiring human review: " f"{validation['review_count']}")
    print(f"Clauses with contractual indicators: " f"{validation['indicator_count']}")
    print(f"Clauses with entities: " f"{validation['entity_count']}")
    print(f"\nValidation errors: " f"{len(validation['errors'])}")

    if validation["errors"]:
        print("\nERRORS:")

        for error in validation["errors"]:
            print(f"- {error}")

        raise SystemExit("\nRisk Intelligence validation FAILED.")

    print("\nRisk Intelligence validation " "PASSED.")


if __name__ == "__main__":
    main()
