"""Lightweight evidence checks for selected high-impact classifier labels."""

import re

# Used only to reject clearly unsupported transformer labels; this does not assign labels.
LABEL_EVIDENCE = {
    "audit rights": ("audit rights", "right to audit", "right to inspect", "books and records", "records access"),
    "cap on liability": (
        "liability cap",
        "cap on liability",
        "maximum liability",
        "liability shall not exceed",
        "aggregate liability",
    ),
    "liquidated damages": ("liquidated damages", "liquidated damage"),
    "uncapped liability": (
        "unlimited liability",
        "uncapped liability",
        "liability without limitation",
        "no limitation on liability",
    ),
    "insurance": ("insurance", "insured", "policy of insurance", "coverage"),
    "covenant not to sue": (
        "covenant not to sue",
        "not sue",
        "waiver of claims",
        "release of claims",
        "release and discharge",
    ),
    "termination for convenience": (
        "terminate for convenience",
        "termination for convenience",
        "without cause",
    ),
    "notice period to terminate renewal": ("notice", "termination", "renewal"),
    "governing law": ("governed by", "governing law", "choice of law"),
    "anti-assignment": ("assign", "assignment", "transfer this agreement"),
    "minimum commitment": (
        "minimum commitment",
        "minimum purchase",
        "committed to purchase",
        "minimum volume",
    ),
    "ip ownership assignment": ("intellectual property", "work product", "ip rights"),
    "license grant": ("license", "licence", "grant to", "right to use"),
    "confidentiality": ("confidential", "non-disclosure"),
    "dispute resolution": (
        "arbitration",
        "dispute",
        "mediation",
        "venue",
        "jurisdiction",
    ),
}


def validate_prediction(text, prediction):
    """Lower confidence for a high-impact CUAD label with no matching text evidence."""
    label = str(prediction.get("clause_type") or "")
    normalized_label = label.casefold()
    anchors = LABEL_EVIDENCE.get(normalized_label)
    if not anchors:
        return {**prediction, "validation": {"status": "not_checked"}}

    folded = " ".join(str(text).casefold().split())
    if normalized_label == "audit rights":
        supported = any(anchor in folded for anchor in anchors) or bool(
            re.search(
                r"\b(?:right|entitled|may|shall)\b.{0,60}\b(?:audit|inspect|inspection)\b|\b(?:audit|inspect|inspection)\b.{0,60}\b(?:records|books|facilities|compliance)\b",
                folded,
            )
        )
    elif normalized_label == "cap on liability":
        supported = bool(
            re.search(
                r"\bliability\b.{0,80}\b(?:cap|maximum|limit|limited|not exceed)\b|\b(?:cap|maximum|limit|limited)\b.{0,80}\bliability\b",
                folded,
            )
        )
    elif normalized_label == "uncapped liability":
        supported = any(anchor in folded for anchor in anchors) or bool(
            re.search(
                r"\bliability\b.{0,60}\b(?:uncapped|unlimited|not limited|not subject to any limitation)\b",
                folded,
            )
        )
    elif normalized_label == "termination for convenience":
        supported = any(anchor in folded for anchor in anchors) or (
            any(term in folded for term in ("termination", "terminate"))
            and any(term in folded for term in ("convenience", "without cause"))
        )
    elif normalized_label == "notice period to terminate renewal":
        supported = "notice" in folded and any(
            term in folded for term in ("termination", "terminate", "renewal", "renew")
        )
    elif normalized_label == "license grant":
        supported = any(term in folded for term in ("license", "licence")) and any(
            term in folded for term in ("grant", "right to use", "permission to use")
        )
    elif normalized_label == "ip ownership assignment":
        supported = any(anchor in folded for anchor in anchors) and any(
            term in folded for term in ("own", "ownership", "assign", "transfer")
        )
    elif normalized_label == "anti-assignment":
        supported = "assignment" in folded or "transfer this agreement" in folded or bool(
            re.search(r"\b(?:may|shall|will|must)(?:\s+not)?\s+assign\b", folded)
        )
    elif normalized_label == "governing law":
        supported = any(anchor in folded for anchor in anchors)
    else:
        supported = any(anchor in folded for anchor in anchors)
    if supported:
        return {**prediction, "validation": {"status": "supported"}}

    original_confidence = float(prediction.get("confidence", 0.0))
    return {
        **prediction,
        "clause_type": "Uncertain classification",
        "confidence": round(min(original_confidence * 0.5, 0.49), 4),
        "validation": {
            "status": "unsupported_label",
            "predicted_clause_type": label,
            "reason": "The clause text does not contain evidence for the predicted category.",
        },
    }
