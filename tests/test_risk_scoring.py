import json

from fastapi.testclient import TestClient

from src.api import main as api
from src.data_processing.clause_processor import extract_clauses
from src.risk.risk_scoring import score_clause, score_contract
from src.risk.risk_engine import find_contractual_phrases


def _assess(text, clause_type="General", confidence=0.92):
    from src.risk.risk_engine import extract_contractual_indicators
    phrases = extract_contractual_indicators(text, [])
    return {
        "clause_type": clause_type,
        "classification_confidence": confidence,
        "clause_text": text,
        "contractual_indicators": phrases,
        "risk_assessment": {"requires_human_review": confidence < .70},
    }


def test_structural_heading_variants_and_fallbacks():
    samples = [
        "Short agreement\n1. Confidentiality. Keep information private.",
        "ARTICLE 1\n1.1 Terms. This is a long enough clause.\nARTICLE II\n2.1 Notices. Give written notice.",
        "Article I\n1. Opening. First provision.\nSECTION 1\nSection text.",
        "SECTION 1\n1.1.1 Nested provision. Body text.\nSection 2\n2. Other provision. More body.",
        "1. Confidentiality. Keep information confidential.\n2. Liability. Each party indemnifies the other.",
        "RECITALS\nWHEREAS the parties agree;\nNOW, THEREFORE the parties promise.",
    ]
    for text in samples:
        clauses = extract_clauses(text)
        assert clauses
        assert all(clause["text"] for clause in clauses)


def test_article_and_section_metadata_are_preserved():
    clauses = extract_clauses("ARTICLE IV\n4.1 Term. Body.\nSECTION 4.2\n4.2.1 Scope. More body.")
    assert clauses[0]["article"] == "IV"
    assert clauses[1]["section"] == "4.2"


def test_risk_scoring_covers_legal_evidence_and_contract_summary():
    texts = [
        ("Confidentiality. Confidential information stays private.", "Confidentiality"),
        ("Indemnification. Supplier shall indemnify and hold harmless.", "Indemnification"),
        ("Limitation of liability. Liability shall not exceed $100,000.", "Limitation of Liability"),
        ("Termination. Either party may terminate with 30 days notice.", "Termination"),
        ("Governing law. Disputes will be resolved by arbitration.", "Governing Law"),
    ]
    clauses = [_assess(text, kind) for text, kind in texts]
    summary = score_contract(clauses)
    assert 0 <= summary["overall_risk_score"] <= 100
    assert summary["overall_risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert all(0 <= c["risk_score"] <= 100 and c["risk_level"] for c in clauses)
    assert summary["high_risk_clause_count"] + summary["low_risk_clause_count"] <= len(clauses)
    json.dumps({"risk_scoring": summary, "clauses": clauses})


def test_short_and_long_contract_scores_are_normalized():
    short = [_assess("Basic notice clause.")]
    long = [_assess(f"Routine provision {i}.") for i in range(40)]
    for clauses in (short, long):
        result = score_contract(clauses)
        assert 0 <= result["overall_risk_score"] <= 100
        assert result["low_risk_clause_count"] + result["high_risk_clause_count"] <= len(clauses)


def test_analyze_api_returns_json_risk_fields(monkeypatch):
    class FakeUpload:
        filename = "contract.pdf"
        async def read(self):
            return b"pdf"

    def fake_pdf(pdf_path, ocr_path):
        ocr_path.write_text("1. Term. Sample clause.", encoding="utf-8")

    result = {
        "contract_report": {"clauses": [], "risk_scoring": {"overall_risk_score": 44, "overall_risk_level": "MEDIUM"}, "risk_summary": {}},
        "review_report": {"review_items": []},
    }
    monkeypatch.setattr(api, "process_pdf", fake_pdf)
    monkeypatch.setattr(api, "run_pipeline", lambda **kwargs: result)
    response = TestClient(api.app).post("/analyze", files={"file": ("contract.pdf", b"pdf", "application/pdf")})
    assert response.status_code == 200
    data = response.json()
    assert data["risk_scoring"]["overall_risk_score"] == 44
    assert json.loads(response.text)["risk_scoring"]["overall_risk_level"] == "MEDIUM"


def test_analyze_api_rejects_empty_upload():
    response = TestClient(api.app).post(
        "/analyze", files={"file": ("empty.pdf", b"", "application/pdf")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded PDF is empty."


def test_analyze_api_returns_clear_error_for_unreadable_pdf():
    response = TestClient(api.app).post(
        "/analyze", files={"file": ("broken.pdf", b"not a PDF", "application/pdf")}
    )
    assert response.status_code == 422
    assert "not a readable PDF" in response.json()["detail"]


def test_common_boilerplate_is_not_mistaken_for_unlimited_liability_or_assignment():
    boilerplate = find_contractual_phrases(
        "The party shall indemnify the other for losses, including, without limitation, reasonable fees, and its successors and assigns."
    )
    assert "unlimited_language" not in boilerplate
    assert "assignment_language" not in boilerplate


def test_unlimited_liability_and_purchase_commitment_remain_detectable():
    evidence = find_contractual_phrases(
        "The supplier's liability is without limitation. Customer is committed to purchase a minimum amount of $500,000."
    )
    assert "unlimited_language" in evidence
    assert "commitment_language" in evidence


def test_no_cap_wording_with_be_is_detected_and_scored_from_its_evidence():
    text = "The Customer's liability shall not exceed any maximum amount and shall not be subject to any limitation."
    phrases = find_contractual_phrases(text)
    assert "unlimited_language" in phrases
    assert "limitation_language" in phrases

    result = score_clause(
        {
            "clause_type": "Uncertain classification",
            "classification_confidence": 0.3411,
            "clause_text": text,
            "contractual_indicators": {"contractual_phrases": phrases},
        }
    )
    assert result["risk_score"] == 70
    assert result["risk_level"] == "HIGH"
    assert "Human review recommended" in result["review_recommendation"]


def test_written_out_noncompete_duration_is_detected_and_explained_correctly():
    from src.risk.risk_engine import find_duration_values

    text = "The Customer shall not compete for three years following termination."
    durations = find_duration_values(text)
    assert durations == ["three years"]

    phrases = find_contractual_phrases(text)
    result = score_clause(
        {
            "clause_type": "Non-Compete",
            "classification_confidence": 0.99,
            "clause_text": text,
            "contractual_indicators": {
                "duration_values": durations,
                "contractual_phrases": phrases,
            },
        }
    )
    assert {factor["factor"] for factor in result["risk_factors"]} >= {
        "Material clause classification: competition restriction",
        "Time-bound contractual term detected",
    }


def test_severe_contract_terms_have_evidence_indicators():
    from src.risk.risk_engine import find_contractual_phrases

    cases = [
        ("The Customer assumes unlimited liability for every loss.", "unlimited_language"),
        ("The Customer shall pay a $1,000,000 penalty without a stated aggregate cap.", "uncapped_penalty_language"),
        ("The Supplier may recover unlimited damages, penalties, and future fees.", "unlimited_language"),
        ("Any unpaid amount incurs a charge of 5% per week, compounded weekly.", "compounding_charge_language"),
        ("The Customer shall defend, indemnify, and hold harmless the Supplier and its affiliates.", "unilateral_indemnity_language"),
        ("The Supplier may terminate immediately for breach; the Customer has no contractual cure period.", "immediate_termination_language"),
        ("Upon termination, the Customer shall pay all remaining fees that would have become due for the entire remaining term.", "fee_acceleration_language"),
        ("The Supplier may amend material terms at any time by posting notice.", "unilateral_change_language"),
        ("The Customer shall not develop or support any service that competes with Supplier.", "competition_language"),
        ("The Supplier alone may select the governing law, venue, and dispute-resolution process.", "unilateral_dispute_language"),
        ("Upon termination, Customer will pay all remaining minimum commitments for the entire five year term.", "fee_acceleration_language"),
    ]
    for text, expected in cases:
        assert expected in find_contractual_phrases(text), (expected, text)


def test_penalty_mention_without_penalty_obligation_is_not_flagged():
    from src.risk.risk_engine import find_contractual_phrases

    phrases = find_contractual_phrases("Customer may terminate without penalty.")
    assert "penalty_language" not in phrases
    assert "uncapped_penalty_language" not in phrases


def test_uncertain_but_explicit_uncapped_liability_receives_evidence_materiality():
    result = score_clause(
        {
            "clause_type": "Uncertain classification",
            "classification_confidence": 0.49,
            "clause_text": "The Customer has unlimited liability.",
            "contractual_indicators": {
                "contractual_phrases": {"unlimited_language": ["unlimited liability"]}
            },
            "risk_assessment": {"requires_human_review": True},
        }
    )
    assert result["risk_score"] == 58
    assert result["risk_level"] == "MEDIUM"
    assert {factor["points"] for factor in result["risk_factors"]} >= {18, 30}
    assert "Human review recommended" in result["review_recommendation"]


def test_low_confidence_affects_review_recommendation_not_severity_score():
    clause = _assess("Routine clause text.", confidence=0.45)
    result = score_clause(clause)
    assert result["risk_score"] == 10
    assert result["risk_level"] == "LOW"
    assert "Human review recommended" in result["review_recommendation"]


def test_score_level_boundaries_are_low_medium_high():
    from src.risk.risk_scoring import _level

    assert [_level(score) for score in (0, 39, 40, 69, 70, 100)] == [
        "LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "HIGH"
    ]


def test_duplicate_indicators_only_contribute_once():
    clause = _assess("Unlimited liability.")
    clause["contractual_indicators"]["contractual_phrases"] = {
        "unlimited_language": ["unlimited liability", "unlimited liability"]
    }
    result = score_clause(clause)
    assert [factor["points"] for factor in result["risk_factors"]].count(30) == 1


def test_empty_and_malformed_score_inputs_are_safe():
    assert score_contract([])["overall_risk_score"] == 0
    assert score_contract([None, {"error": "inference failed"}])["overall_risk_score"] == 0
    malformed = score_clause({"classification_confidence": "not-a-number", "contractual_indicators": []})
    assert malformed["risk_score"] == 10
    assert malformed["risk_level"] == "LOW"


def test_invalid_clause_object_has_clear_error():
    import pytest

    with pytest.raises(ValueError, match="Clause data must be an object"):
        score_clause(None)
