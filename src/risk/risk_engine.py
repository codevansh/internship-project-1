import re
from typing import Dict, List

CONFIDENCE_THRESHOLD = 0.70


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
    """Extract common contractual durations."""
    return re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:day|days|week|weeks|month|months|year|years)\b",
        text,
        flags=re.IGNORECASE,
    )


def find_contractual_phrases(text: str) -> Dict[str, List[str]]:
    """
    Detect contractual language patterns from the clause text.

    These are evidence indicators and are not legal-risk classifications.
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
        "indemnification_language": ["indemnify", "indemnification", "hold harmless"],
        "confidentiality_language": ["confidential information", "confidentiality", "non-disclosure"],
        "dispute_language": ["arbitration", "dispute resolution", "governing law", "venue", "jurisdiction"],
    }

    text_lower = text.lower()
    detected = {}

    for category, phrases in phrase_groups.items():
        matches = []
        for phrase in phrases:
            if phrase.lower() not in text_lower:
                continue
            if phrase == "without limitation":
                for occurrence in re.finditer(re.escape(phrase), text_lower):
                    prefix = text_lower[max(0, occurrence.start() - 35):occurrence.start()]
                    context = text_lower[max(0, occurrence.start() - 45):occurrence.end() + 55]
                    if re.search(r"\bincluding\s*,?\s*$", prefix):
                        continue
                    if not re.search(r"\b(?:liability|obligation|responsibility|claim|amount|damages)\b", context):
                        continue
                    matches.append(phrase)
                    break
            elif phrase == "minimum amount":
                if re.search(r"(?:\b(?:purchase|quantity|volume|commit)\w*.{0,60}\bminimum amount\b|\bminimum amount\b.{0,60}\b(?:purchase|quantity|volume|commit)\w*)", text_lower):
                    matches.append(phrase)
            elif phrase == "assign":
                if re.search(r"\b(?:may|shall|will|must|may not|shall not)\s+assign\b|\bassign (?:its|their|this|the)\b", text_lower):
                    matches.append(phrase)
            elif phrase in {"no limit", "not subject to any limitation", "without any limitation"}:
                if re.search(r"\b(?:liability|obligation|responsibility|damages)\b", text_lower):
                    matches.append(phrase)
            else:
                matches.append(phrase)

        if matches:
            detected[category] = matches

    return detected


def extract_contractual_indicators(
    clause_text: str,
    entities: List[Dict],
) -> Dict:
    """Extract observable contractual attributes from a clause.
    This function does not assign legal risk"""

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


def assess_risk(
    classification_confidence: float,
    indicators: Dict,
) -> Dict:
    """
    Assess the level of contractual attention suggested by
    observable indicators and model confidence.

    This is an evidence-based screening signal, not a legal
    determination of whether a clause is legally risky.
    """

    reasons = []

    phrase_groups = indicators.get(
        "contractual_phrases",
        {},
    )

    if phrase_groups.get("unlimited_language"):
        reasons.append("Unlimited or uncapped liability-related language detected.")

    if phrase_groups.get("limitation_language"):
        reasons.append("Liability or obligation limitation language detected.")

    if phrase_groups.get("termination_language"):
        reasons.append("Termination-related language detected.")

    if phrase_groups.get("assignment_language"):
        reasons.append("Assignment or transfer language detected.")

    if phrase_groups.get("competition_language"):
        reasons.append("Competition-restriction language detected.")

    if phrase_groups.get("commitment_language"):
        reasons.append("Minimum or committed purchase language detected.")

    if phrase_groups.get("damages_language"):
        reasons.append("Damages-related language detected.")

    if indicators.get("monetary_values"):
        reasons.append("Monetary values detected in the clause.")

    if indicators.get("percentage_values"):
        reasons.append("Percentage-based contractual values detected.")

    if indicators.get("duration_values"):
        reasons.append("Contractual duration values detected.")

    if classification_confidence < CONFIDENCE_THRESHOLD:
        reasons.append(
            "Clause classification confidence is below the human-review threshold."
        )

    if not reasons:
        level = "LOW"
    elif classification_confidence < CONFIDENCE_THRESHOLD:
        level = "REVIEW"
    elif len(reasons) >= 3:
        level = "ATTENTION"
    else:
        level = "INDICATOR"

    return {
        "level": level,
        "reasons": reasons,
        "evidence": indicators,
        "requires_human_review": (classification_confidence < CONFIDENCE_THRESHOLD),
    }


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
    This layer extracts contractual indicators and
    produces an evidence-based screening signal.
    """

    indicators = extract_contractual_indicators(
        clause_text,
        entities,
    )

    risk_assessment = assess_risk(
        classification_confidence=classification_confidence,
        indicators=indicators,
    )

    return {
        "clause_type": clause_type,
        "classification_confidence": classification_confidence,
        "clause_text": clause_text,
        "entities": entities,
        "contractual_indicators": indicators,
        "risk_assessment": risk_assessment,
    }
