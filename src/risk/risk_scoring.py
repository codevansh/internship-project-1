"""Transparent, deterministic screening scores built from pipeline evidence."""

from collections import Counter
import math
from typing import Dict, List


def _level(score: int) -> str:
    return "LOW" if score < 40 else "MEDIUM" if score < 70 else "HIGH"


def score_clause(clause: Dict) -> Dict:
    """Score a classifier/indicator result; no legal conclusion is implied."""
    if not isinstance(clause, dict):
        raise ValueError("Clause data must be an object.")
    score = 10
    factors = []
    clause_type = str(clause.get("clause_type") or "").lower().replace("_", " ")
    confidence = clause.get("classification_confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError, OverflowError):
        confidence = None
    if confidence is not None and (not math.isfinite(confidence) or not 0 <= confidence <= 1):
        confidence = None
    indicators = clause.get("contractual_indicators") or {}
    if not isinstance(indicators, dict):
        indicators = {}
    phrases = indicators.get("contractual_phrases") or {}
    if not isinstance(phrases, dict):
        phrases = {}

    # Clause class provides a modest materiality prior; observed evidence does the rest.
    severity_groups = {
        "liability / indemnity": (
            ("liability", "indemnif", "damages", "insurance"),
            ("unlimited_language", "damages_language", "indemnification_language", "unilateral_indemnity_language", "uncapped_penalty_language"),
            18,
        ),
        "termination / renewal": (
            ("termination", "renewal", "default"),
            ("termination_language", "immediate_termination_language", "fee_acceleration_language"),
            12,
        ),
        "competition restriction": (
            ("non-compete", "noncompete", "competition restriction"),
            ("competition_language",),
            12,
        ),
        "confidentiality / data": (
            ("confidential", "privacy", "data protection", "security"),
            ("confidentiality_language",),
            10,
        ),
        "dispute / governing law": (
            ("governing law", "dispute", "arbitration", "jurisdiction"),
            ("dispute_language", "unilateral_dispute_language"),
            10,
        ),
        "payment / commitment": (
            ("payment", "price", "purchase", "commitment"),
            ("commitment_language", "penalty_language", "compounding_charge_language", "uncapped_penalty_language"),
            8,
        ),
        "unilateral contract terms": (
            (),
            ("unilateral_change_language",),
            8,
        ),
    }
    # Prefer the model's explicit clause category. Otherwise, use one
    # evidence-based group as a modest prior.
    materiality = next(
        (
            (name, weight)
            for name, (terms, _evidence_keys, weight) in severity_groups.items()
            if any(term in clause_type for term in terms)
        ),
        None,
    )
    if materiality is None:
        materiality = next(
            (
                (name, weight)
                for name, (_terms, evidence_keys, weight) in severity_groups.items()
                if any(phrases.get(key) for key in evidence_keys)
            ),
            None,
        )
    if materiality:
        name, weight = materiality
        score += weight
        factors.append(
            {"factor": f"Material clause classification: {name}", "points": weight}
        )

    phrase_weights = {
        "unlimited_language": (
            30,
            "Uncapped or unlimited obligation language detected",
        ),
        "limitation_language": (
            12,
            "Limitation or cap language detected; scope should be reviewed",
        ),
        "damages_language": (15, "Damages language detected"),
        "penalty_language": (20, "Penalty or default-charge language detected"),
        "compounding_charge_language": (15, "Compounding charge language detected"),
        "uncapped_penalty_language": (18, "Uncapped penalty or damages language detected"),
        "immediate_termination_language": (15, "Termination without a cure opportunity detected"),
        "fee_acceleration_language": (18, "Future-fee acceleration language detected"),
        "unilateral_change_language": (15, "Unilateral contract-change language detected"),
        "unilateral_indemnity_language": (12, "Customer-side indemnity obligation detected"),
        "unilateral_dispute_language": (15, "Supplier-controlled dispute terms detected"),
        "competition_language": (15, "Competition restriction language detected"),
        "commitment_language": (12, "Minimum or committed purchase language detected"),
        "termination_language": (5, "Termination language detected"),
        "assignment_language": (5, "Assignment or transfer language detected"),
        "indemnification_language": (18, "Indemnification language detected"),
        "confidentiality_language": (8, "Confidentiality obligation detected"),
        "dispute_language": (10, "Dispute resolution language detected"),
    }
    for key, (weight, description) in phrase_weights.items():
        if phrases.get(key):
            score += weight
            factors.append({"factor": description, "points": weight})

    if indicators.get("monetary_values") or indicators.get("percentage_values"):
        score += 5
        factors.append(
            {"factor": "Quantified financial or percentage term detected", "points": 5}
        )
    if indicators.get("duration_values"):
        score += 3
        factors.append({"factor": "Time-bound contractual term detected", "points": 3})

    risk_assessment = clause.get("risk_assessment") or {}
    if not isinstance(risk_assessment, dict):
        risk_assessment = {}
    requires_review = bool(risk_assessment.get("requires_human_review"))
    if confidence is not None and confidence < 0.70:
        requires_review = True
    score = min(100, max(0, round(score)))
    return {
        "risk_score": score,
        "risk_level": _level(score),
        "risk_factors": factors,
        "explanation": (
            "Screening score combines clause classification, detected contractual indicators, and quantified terms. "
            + (
                "Applied factors: " + "; ".join(f["factor"] for f in factors) + "."
                if factors
                else "No configured risk factors were detected beyond the baseline score."
            )
        ),
        "evidence": clause.get("clause_text", ""),
        "article": clause.get("article"),
        "article_title": clause.get("article_title"),
        "section": clause.get("section"),
        "section_title": clause.get("section_title"),
        "schedule": clause.get("schedule"),
        "clause_number": clause.get("clause_number"),
        "title": clause.get("title"),
        "confidence": confidence,
        "review_recommendation": (
            "Human review recommended because classification confidence is low or the risk engine flagged review."
            if requires_review
            else "No confidence-based human-review flag was raised."
        ),
    }


def score_contract(clauses: List[Dict]) -> Dict:
    """Weighted mean plus a high-risk tail premium, normalized to 0–100."""
    scored = []
    for clause in clauses:
        if not isinstance(clause, dict):
            continue
        if "error" in clause:
            continue
        scoring = score_clause(clause)
        clause.update(scoring)
        scored.append(clause)
    if not scored:
        return {
            "overall_risk_score": 0,
            "overall_risk_level": "LOW",
            "high_risk_clause_count": 0,
            "review_clause_count": 0,
            "medium_risk_clause_count": 0,
            "human_review_clause_count": 0,
            "low_risk_clause_count": 0,
            "major_risk_factors": [],
            "risk_summary": "No clauses were successfully assessed.",
            "scoring_method": "Weighted clause mean plus high-risk tail premium.",
        }

    weights = [
        (
            1.75
            if c["risk_level"] == "HIGH"
            else 1.2 if c["risk_level"] == "MEDIUM" else 1.0
        )
        for c in scored
    ]
    weighted_mean = sum(c["risk_score"] * w for c, w in zip(scored, weights)) / sum(
        weights
    )
    high_count = sum(c["risk_level"] == "HIGH" for c in scored)
    # A bounded premium makes serious clauses more visible while keeping the result normalized.
    tail_premium = min(20, high_count * 5)
    overall = min(100, round(weighted_mean + tail_premium))
    medium_count = sum(c["risk_level"] == "MEDIUM" for c in scored)
    review_count = 0
    for clause in scored:
        assessment = clause.get("risk_assessment") or {}
        validation = clause.get("classification_validation") or {}
        confidence = clause.get("confidence")
        review_count += bool(
            (isinstance(assessment, dict) and assessment.get("requires_human_review"))
            or (isinstance(validation, dict) and validation.get("status") == "unsupported_label")
            or (isinstance(confidence, float) and math.isfinite(confidence) and confidence < 0.70)
        )
    factor_counts = Counter(f["factor"] for c in scored for f in c["risk_factors"])
    major = [name for name, _ in factor_counts.most_common(5)]
    return {
        "overall_risk_score": overall,
        "overall_risk_level": _level(overall),
        "high_risk_clause_count": high_count,
        # Kept as the legacy alias used by older clients. It means medium risk,
        # while human_review_clause_count is the separate review queue count.
        "review_clause_count": medium_count,
        "medium_risk_clause_count": medium_count,
        "human_review_clause_count": review_count,
        "low_risk_clause_count": sum(c["risk_level"] == "LOW" for c in scored),
        "major_risk_factors": major,
        "risk_summary": f"{_level(overall)} screening risk across {len(scored)} assessed clauses; {high_count} high-risk clauses and {review_count} clauses flagged for review.",
        "scoring_method": "Weighted mean (HIGH clauses 1.75×, MEDIUM 1.2×, LOW 1×) plus 5 points per HIGH clause, capped at a 20-point premium; final score capped at 100.",
    }
