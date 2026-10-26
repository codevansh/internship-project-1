import re
from typing import Dict, List

CONFIDENCE_THRESHOLD = 0.70

NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|"
    "twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety"
)
DURATION_NUMBER = (
    rf"(?:\d+(?:\.\d+)?|(?:{NUMBER_WORDS})(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?)"
)


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
    """Extract numeric and written-out contractual durations."""
    return re.findall(
        rf"\b{DURATION_NUMBER}\s*-?\s*(?:days?|weeks?|months?|years?)\b",
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
            "not be subject to any limitation",
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
            elif phrase in {
                "no limit",
                "not subject to any limitation",
                "not be subject to any limitation",
                "without any limitation",
            }:
                if re.search(r"\b(?:liability|obligation|responsibility|damages)\b", text_lower):
                    matches.append(phrase)
            else:
                matches.append(phrase)

        if matches:
            detected[category] = matches

    # Cover common word-order variations while tying uncapped wording to an
    # obligation. This avoids treating boilerplate "including, without limitation"
    # as evidence that liability itself is uncapped.
    if re.search(
        r"\b(?:unlimited|uncapped)\b.{0,60}\b(?:customer\s+)?(?:liabilit\w*|obligations?|responsibilit\w*|damages|losses|claims|penalt\w*|fees?|amounts?)\b|"
        r"\b(?:liabilit\w*|obligations?|responsibilit\w*|damages|losses|claims|penalt\w*|fees?|amounts?)\b.{0,60}\b(?:unlimited|uncapped)\b",
        text_lower,
    ):
        detected.setdefault("unlimited_language", []).append("unlimited obligation")

    evidence_patterns = {
        "penalty_language": (
            r"\b(?:shall|must|will|agrees? to)\s+pay\b.{0,80}\bpenalt(?:y|ies)\b|"
            r"\bpenalt(?:y|ies)\s+(?:of|in the amount of)\s+\$?[\d,]+|"
            r"\b(?:late|default)\s+(?:fees?|charges?|interest)\b|"
            r"\b(?:charge|interest)\s+of\s+\d+(?:\.\d+)?\s*%",
            "contractual penalty or default charge",
        ),
        "compounding_charge_language": (
            r"\bcompound(?:ed|ing|s)\b.{0,35}\b(?:daily|weekly|monthly|quarterly|annually|interest|charges?)\b|"
            r"\b(?:daily|weekly|monthly|quarterly)\s+compounding\b",
            "compounding charge",
        ),
        "uncapped_penalty_language": (
            r"\b(?:penalt(?:y|ies)|damages|default charges?)\b.{0,120}\b(?:uncapped|unlimited|without\s+(?:a\s+)?(?:stated\s+)?(?:aggregate\s+)?cap|not subject to (?:an? )?aggregate cap)\b|"
            r"\b(?:uncapped|unlimited)\b.{0,60}\b(?:penalt(?:y|ies)|damages|default charges?)\b",
            "uncapped penalty or damages obligation",
        ),
        "immediate_termination_language": (
            r"\bterminate\w*\b.{0,60}\bimmediately\b|\bimmediately\b.{0,60}\bterminate\w*\b|"
            r"\bno\s+(?:contractual\s+)?cure\s+period\b|\bwithout\s+(?:an?\s+)?cure\s+period\b|"
            r"\bwithout\s+(?:prior\s+)?notice\b.{0,50}\b(?:terminate|termination)\b",
            "immediate termination or no cure period",
        ),
        "fee_acceleration_language": (
            r"\b(?:accelerat\w*\s+(?:all\s+)?(?:future\s+|remaining\s+)?fees?|"
            r"(?:all|remaining|future)\s+(?:minimum\s+)?(?:fees?|commitments?)\b.{0,140}\b(?:remaining term|entire(?:\s+\w+){0,3}\s+term|would have become due|immediately due))\b",
            "accelerated future fees or remaining-term commitments",
        ),
        "unilateral_change_language": (
            r"\b(?:supplier|service provider|provider|vendor|seller|licensor)\b.{0,100}\b(?:may|can|is entitled to)\b.{0,30}\b(?:unilaterally\s+)?(?:amend|modify|change|revise)\b|"
            r"\b(?:amend|modify|change|revise)\b.{0,100}\b(?:at its sole discretion|without (?:prior )?(?:written )?consent)\b",
            "one-sided right to change contract terms",
        ),
        "unilateral_indemnity_language": (
            r"\b(?:customer|client|buyer|licensee)\b.{0,100}\b(?:defend\s*,?\s*)?indemnif(?:y|ies)\b.{0,180}\b(?:supplier|service provider|provider|vendor|seller|licensor)\b",
            "customer-side indemnity in favor of supplier",
        ),
        "unilateral_dispute_language": (
            r"\b(?:supplier|service provider|provider|vendor|seller|licensor)\b.{0,100}\b(?:alone|solely|at its sole discretion)\b.{0,100}\b(?:select|choose|designate|determine)\b.{0,100}\b(?:governing law|venue|forum|dispute(?:[- ]resolution| process)?)\b",
            "supplier-controlled law, venue, or dispute process",
        ),
    }
    for key, (pattern, evidence) in evidence_patterns.items():
        if re.search(pattern, text_lower):
            detected.setdefault(key, []).append(evidence)

    if re.search(r"\bshall\s+not\b.{0,160}\bcompetes?\b", text_lower):
        detected.setdefault("competition_language", []).append(
            "prohibition on developing or supporting competing products"
        )
    if re.search(r"\b(?:purchase|purchases?)\b.{0,50}\bno less than\b|\bno less than\b.{0,50}\b(?:purchase|quantity|volume)\b", text_lower):
        detected.setdefault("commitment_language", []).append("no-less-than purchase obligation")

    # Normalize repeated matches from exact phrase and contextual detectors.
    for matches in detected.values():
        matches[:] = list(dict.fromkeys(matches))

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

    for key, description in (
        ("penalty_language", "Penalty or default-charge language detected."),
        ("immediate_termination_language", "Termination without a cure opportunity detected."),
        ("fee_acceleration_language", "Future-fee acceleration language detected."),
        ("unilateral_change_language", "Unilateral contract-change language detected."),
        ("unilateral_indemnity_language", "Customer-side indemnity obligation detected."),
        ("compounding_charge_language", "Compounding charge language detected."),
        ("uncapped_penalty_language", "Uncapped penalty or damages language detected."),
        ("unilateral_dispute_language", "Supplier-controlled dispute terms detected."),
    ):
        if phrase_groups.get(key):
            reasons.append(description)

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
