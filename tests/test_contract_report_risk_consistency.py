from src.risk.contract_report import build_contract_report, build_contract_risk_summary
from src.risk.review_report import build_review_report
from src.risk.risk_scoring import score_contract


def _clause(
    number,
    *,
    clause_type="General",
    confidence=0.95,
    phrases=None,
    assessment_level="LOW",
    requires_review=False,
):
    return {
        "article": "1",
        "clause_number": number,
        "title": f"Clause {number}",
        "clause_type": clause_type,
        "classification_confidence": confidence,
        "clause_text": f"Evidence for clause {number}.",
        "contractual_indicators": {"contractual_phrases": phrases or {}},
        "risk_assessment": {
            "level": assessment_level,
            "reasons": [assessment_level],
            "requires_human_review": requires_review,
        },
    }


def _summarize(clauses):
    scoring = score_contract(clauses)
    report = build_contract_report(clauses)
    review = build_review_report(clauses)
    summary = build_contract_risk_summary(report, review)
    summary.update(scoring)
    summary["risk_score"] = scoring["overall_risk_score"]
    return summary, review


def _high_clause(number):
    return _clause(
        number,
        clause_type="Uncapped Liability",
        phrases={
            "unlimited_language": ["unlimited liability"],
            "damages_language": ["damages"],
            "indemnification_language": ["indemnification"],
        },
    )


def test_score_based_high_clause_list_includes_high_clause_without_attention():
    summary, _ = _summarize([_high_clause("1.1")])
    assert summary["high_risk_clause_count"] == 1
    assert len(summary["high_risk_clauses"]) == 1
    assert summary["high_risk_clauses"][0]["clause_number"] == "1.1"
    assert summary["high_risk_clauses"][0]["risk_level"] == "HIGH"
    assert summary["high_risk_clauses"][0]["level"] == "HIGH"
    assert summary["risk_score"] == summary["overall_risk_score"]
    assert summary["attention_clauses"] == []


def test_attention_clause_below_high_threshold_is_separate():
    summary, _ = _summarize([_clause("1.2", assessment_level="ATTENTION")])
    assert summary["high_risk_clause_count"] == 0
    assert summary["high_risk_clauses"] == []
    assert len(summary["attention_clauses"]) == 1
    assert summary["attention_clauses"][0]["clause_number"] == "1.2"


def test_multiple_high_clause_count_matches_clause_list_and_metadata():
    summary, _ = _summarize([_high_clause("1.1"), _high_clause("1.2")])
    assert summary["high_risk_clause_count"] == len(summary["high_risk_clauses"]) == 2
    assert [item["clause_number"] for item in summary["high_risk_clauses"]] == ["1.1", "1.2"]
    assert all(item["risk_score"] >= 70 and item["risk_level"] == "HIGH" for item in summary["high_risk_clauses"])


def test_no_high_clauses_returns_empty_high_risk_list():
    summary, _ = _summarize([_clause("1.3")])
    assert summary["high_risk_clause_count"] == 0
    assert summary["high_risk_clauses"] == []


def test_human_review_clause_can_remain_below_high_risk():
    clause = _clause("1.4", confidence=0.55, requires_review=True)
    summary, review = _summarize([clause])
    assert summary["high_risk_clause_count"] == 0
    assert summary["high_risk_clauses"] == []
    assert summary["human_review_clause_count"] == 1
    assert review["review_summary"]["clauses_requiring_review"] == 1
