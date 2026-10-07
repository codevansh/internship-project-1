from src.model.classification_validation import validate_prediction
from src.risk.risk_scoring import score_clause, score_contract


def test_semantically_unsupported_high_impact_labels_become_uncertain():
    cases = [
        ("The customer may set off amounts owed against fees.", "Audit Rights"),
        (
            "Supplier shall notify Customer of a security incident.",
            "Covenant Not To Sue",
        ),
        (
            "Personal information must be protected under applicable law.",
            "Audit Rights",
        ),
        ("Changes must be signed by both parties.", "Covenant Not To Sue"),
        ("Monetary damages may not provide an adequate remedy.", "Liquidated Damages"),
        ("Supplier disclaims all warranties of merchantability.", "Cap On Liability"),
        ("Including, without limitation, reasonable attorneys fees.", "Uncapped Liability"),
        ("The annual audit report is attached.", "Audit Rights"),
    ]
    for text, label in cases:
        result = validate_prediction(text, {"clause_type": label, "confidence": 0.94})
        assert result["clause_type"] == "Uncertain classification"
        assert result["confidence"] <= 0.49
        assert result["validation"]["status"] == "unsupported_label"


def test_supported_prediction_remains_model_assigned():
    result = validate_prediction(
        "Supplier shall maintain insurance coverage during the term.",
        {"clause_type": "Insurance", "confidence": 0.88},
    )
    assert result["clause_type"] == "Insurance"
    assert result["confidence"] == 0.88
    assert result["validation"]["status"] == "supported"


def test_supported_liability_cap_requires_actual_liability_limit_language():
    result = validate_prediction(
        "Aggregate liability shall not exceed $1,000,000.",
        {"clause_type": "Cap On Liability", "confidence": 0.83},
    )
    assert result["validation"]["status"] == "supported"


def test_uncertain_label_does_not_add_category_materiality_prior():
    scored = score_clause(
        {
            "clause_type": "Uncertain classification",
            "classification_confidence": 0.45,
            "clause_text": "The parties amend the agreement in writing.",
            "contractual_indicators": {},
            "risk_assessment": {"requires_human_review": True},
        }
    )
    assert scored["risk_score"] == 25
    assert scored["risk_level"] == "LOW"


def test_score_levels_and_overall_aggregation_stay_bounded():
    clauses = [
        {
            "clause_type": "General",
            "classification_confidence": 0.95,
            "clause_text": "routine",
            "contractual_indicators": {},
        },
        {
            "clause_type": "Uncapped Liability",
            "classification_confidence": 0.95,
            "clause_text": "unlimited liability, damages, and indemnification; $500,000.",
            "contractual_indicators": {
                "monetary_values": ["$500,000"],
                "contractual_phrases": {
                    "unlimited_language": ["unlimited"],
                    "damages_language": ["damages"],
                    "indemnification_language": ["indemnification"],
                },
            },
        },
    ]
    summary = score_contract(clauses)
    assert clauses[0]["risk_level"] == "LOW"
    assert clauses[1]["risk_level"] == "HIGH"
    assert 0 <= summary["overall_risk_score"] <= 100
    assert summary["high_risk_clause_count"] == 1
