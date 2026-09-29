import re
from typing import Dict, List


def find_money_entities(entities: List[Dict]) -> List[Dict]:
    """Return MONEY entities extracted by spaCy."""
    return [entity for entity in entities if entity.get("label") == "MONEY"]


def find_date_entities(entities: List[Dict]) -> List[Dict]:
    """Return DATE entities extracted by spaCy."""
    return [entity for entity in entities if entity.get("label") == "DATE"]


def find_percentage_values(text: str) -> List[str]:
    """Extract percentage values directly from the clause text."""
    return re.findall(
        r"\b\d+(?:\.\d+)?\s*%",
        text,
    )


def find_duration_values(text: str) -> List[str]:
    """
    Extract common contractual durations such as:
    30 days, 12 months, 2 years.
    """
    matches = re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:day|days|week|weeks|month|months|year|years)\b",
        text,
        flags=re.IGNORECASE,
    )

    return matches


def find_contractual_phrases(text: str) -> Dict[str, List[str]]:
    """
    Detect generic contractual phrases from the actual clause text.
    These are evidence indicators, not legal-risk classifications.
    """
    phrase_groups = {
        "limitation_language": [
            "shall not exceed",
            "not exceed",
            "limited to",
            "subject to a maximum",
            "maximum liability",
            "liability cap",
        ],
        "unlimited_language": [
            "unlimited liability",
            "without limitation",
            "without any limitation",
            "not subject to any limitation",
            "no limit",
            "uncapped",
        ],
        "termination_language": [
            "terminate",
            "termination",
            "termination notice",
        ],
        "assignment_language": [
            "assign",
            "assignment",
            "transfer",
        ],
        "competition_language": [
            "non-compete",
            "noncompete",
            "shall not compete",
            "not compete",
        ],
        "commitment_language": [
            "minimum commitment",
            "minimum purchase",
            "minimum amount",
            "minimum quantity",
            "committed to purchase",
        ],
        "damages_language": [
            "liquidated damages",
            "damages",
        ],
    }

    text_lower = text.lower()
    detected = {}

    for category, phrases in phrase_groups.items():
        matches = [phrase for phrase in phrases if phrase.lower() in text_lower]

        if matches:
            detected[category] = matches

    return detected


def extract_contractual_indicators(
    clause_text: str,
    entities: List[Dict],
) -> Dict:
    """
    Extract observable contractual attributes from a clause.

    This function does not assign a legal-risk score.
    It reports evidence found in the actual text.
    """

    money_entities = find_money_entities(entities)
    date_entities = find_date_entities(entities)

    percentages = find_percentage_values(clause_text)
    durations = find_duration_values(clause_text)
    phrases = find_contractual_phrases(clause_text)

    indicators = {}

    if money_entities:
        indicators["monetary_values"] = [entity["text"] for entity in money_entities]

    if date_entities:
        indicators["date_entities"] = [entity["text"] for entity in date_entities]

    if percentages:
        indicators["percentage_values"] = percentages

    if durations:
        indicators["duration_values"] = durations

    if phrases:
        indicators["contractual_phrases"] = phrases

    return indicators


def analyze_risk(
    clause_type: str,
    classification_confidence: float,
    clause_text: str,
    entities: List[Dict],
) -> Dict:
    """
    Build a structured contractual intelligence result.

    Legal-RoBERTa supplies the clause type and confidence.
    spaCy supplies named entities.
    This layer extracts additional contractual indicators.

    No learned or universal legal-risk score is produced.
    """

    indicators = extract_contractual_indicators(
        clause_text,
        entities,
    )

    return {
        "clause_type": clause_type,
        "classification_confidence": classification_confidence,
        "clause_text": clause_text,
        "entities": entities,
        "contractual_indicators": indicators,
    }
