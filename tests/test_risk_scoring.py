import json

from fastapi.testclient import TestClient

from src.api import main as api
from src.data_processing.clause_processor import extract_clauses
from src.risk.risk_scoring import score_contract
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
