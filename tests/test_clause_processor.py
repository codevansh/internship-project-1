from src.data_processing.clause_processor import extract_clauses


def test_article_is_derived_from_clause_number():
    ocr_text = """
    ARTICLE 4

    4.1 Term.
    This agreement begins on January 1, 2020.

    Articles

    Representations and Warranties; Indeminification

    5.1 Representations and Warranties.
    Each party represents and warrants that it has authority to enter this agreement.

    5.2 Indemnification.
    Each party shall indemnify the other party against certain claims.

    ARTICLE 6

    6.1 Notices.
    Notices shall be provided in writing.
    """

    clauses = extract_clauses(ocr_text)

    clause_map = {clause["clause_number"]: clause["article"] for clause in clauses}

    assert clause_map["4.1"] == "4"
    assert clause_map["5.1"] == "5"
    assert clause_map["5.2"] == "5"
    assert clause_map["6.1"] == "6"


def test_article_section_and_numbered_heading_variants_preserve_hierarchy():
    text = """ARTICLE I — INTERPRETATION
1.1 Definitions
In this Agreement, defined terms have the meanings stated below.
1.2 Interpretation
Words in the singular include the plural.
ARTICLE 2 - TERM
SECTION 2.1
2.1 Term
The term begins on the Effective Date.
ARTICLE 7
7.5 Termination
Either party may terminate this Agreement on written notice.
7.6 Remedies
The parties may seek equitable relief.
7.7 Disputes
Disputes shall be resolved by arbitration.
7.8 Arbitration
The arbitration shall take place in New York.
8.13 Confidentiality Covenant
Each party shall protect Confidential Information.
"""
    clauses = extract_clauses(text)
    by_number = {clause["clause_number"]: clause for clause in clauses}
    assert by_number["1.1"]["article"] == "I"
    assert by_number["1.2"]["title"] == "Interpretation"
    assert by_number["2.1"]["section"] == "2.1"
    assert by_number["7.5"]["title"] == "Termination"
    assert by_number["7.7"]["text"] == "Disputes shall be resolved by arbitration."
    assert by_number["8.13"]["title"] == "Confidentiality Covenant"


def test_recitals_and_whereas_are_preserved():
    clauses = extract_clauses("RECITALS\nWHEREAS the parties desire to cooperate;\nNOW, THEREFORE, the parties agree.")
    assert len(clauses) == 3
    assert clauses[1]["text"].startswith("WHEREAS")


def test_document_furniture_and_ocr_scraps_are_filtered_but_short_clause_kept():
    text = """MASTER SERVICES AGREEMENT
Page 1
TABLE OF CONTENTS
7.5 Termination .................. 12
Lakeside Boulevard
Lakeside Boulevard 250 Williams Street, Suite 250
LEGAL_1:000001
***
dee
MASTER SERVICES AGREEMENT
1.1 Notice
Give notice.
Pay.
"""
    clauses = extract_clauses(text)
    combined = " ".join(clause["text"] for clause in clauses)
    assert all(fragment not in combined for fragment in ("Boulevard", "LEGAL_1", "***", "dee", "TABLE OF CONTENTS"))
    assert any(clause["clause_number"] == "1.1" and "Give notice." in clause["text"] for clause in clauses)
    assert any("Pay." in clause["text"] for clause in clauses)


def test_provisionless_introduction_and_evaluation_material_are_excluded():
    text = """This synthetic example is provided for evaluation only.
Please use the test checklist to grade the contract.
ARTICLE 1 - LIABILITY
1.1 Unlimited liability. Supplier accepts unlimited liability.
Evaluation checklist: confirm the expected score is high.
Page 1
"""
    clauses = extract_clauses(text)
    assert len(clauses) == 1
    assert clauses[0]["clause_number"] == "1.1"
    assert "evaluation" not in clauses[0]["text"].lower()


def test_synthetic_data_clause_is_preserved_verbatim():
    text = "ARTICLE 1 - DATA\n1.1 Synthetic Data\nSupplier may process synthetic data only as authorized by Customer."
    clauses = extract_clauses(text)
    assert clauses[0]["title"] == "Synthetic Data"
    assert clauses[0]["text"] == "Supplier may process synthetic data only as authorized by Customer."


def test_legitimate_unnumbered_provision_before_first_article_is_retained():
    provision = "The parties shall keep the terms of this Agreement confidential."
    clauses = extract_clauses(f"{provision}\nARTICLE 1 - SERVICES\n1.1 Services. Supplier shall provide support.")
    assert clauses[0]["clause_number"] == "U1"
    assert clauses[0]["text"] == provision
    assert clauses[0]["extraction_note"]
    assert clauses[1]["clause_number"] == "1.1"


def test_identifiable_document_overview_is_not_emitted_as_a_clause():
    text = "This document provides an overview of the proposed services.\nARTICLE 1 - SERVICES\n1.1 Services. Supplier shall provide support."
    clauses = extract_clauses(text)
    assert len(clauses) == 1
    assert clauses[0]["clause_number"] == "1.1"


def test_retained_front_matter_keeps_source_order_and_wording():
    first = "The Customer shall pay invoices within thirty days."
    second = "Supplier shall protect Customer Data."
    clauses = extract_clauses(f"{first}\n{second}\nARTICLE 1 - TERM\n1.1 Term. This Agreement begins today.")
    assert [clause["text"] for clause in clauses] == [
        f"{first} {second}",
        "This Agreement begins today.",
    ]


def test_inline_page_artifact_is_removed_without_losing_provision_text():
    text = "1.1 Remedy. Customer may recover all direct damages. Synthetic QA test artifact — not a real contract Page 1"
    clauses = extract_clauses(text)
    assert clauses[0]["text"] == "Customer may recover all direct damages."


def test_standalone_synthetic_qa_disclaimer_is_not_emitted():
    clauses = extract_clauses(
        "Synthetic QA artifact for testing clause scoring.\n"
        "ARTICLE 1 - SERVICES\n1.1 Services\nSupplier shall provide support."
    )
    assert len(clauses) == 1
    assert clauses[0]["clause_number"] == "1.1"
    assert "Synthetic QA" not in clauses[0]["text"]


def test_inline_article_heading_does_not_contaminate_preceding_clause():
    text = "1.1 Services. Supplier shall provide support. ARTICLE 2 - PAYMENT\n2.1 Fees\nCustomer shall pay the stated fee."
    clauses = extract_clauses(text)
    assert [clause["clause_number"] for clause in clauses] == ["1.1", "2.1"]
    assert clauses[0]["text"] == "Supplier shall provide support."
    assert clauses[1]["article"] == "2"
    assert clauses[1]["text"] == "Customer shall pay the stated fee."


def test_toc_dotted_entries_are_not_clauses():
    clauses = extract_clauses("CONTENTS\nARTICLE 1 — TERM .......... 2\n1.1 Renewal .......... 3\n1.2 Notices 4\nARTICLE 1\n1.1 Renewal\nThe Agreement renews annually.")
    assert len(clauses) == 1
    assert clauses[0]["clause_number"] == "1.1"
    assert "renews annually" in clauses[0]["text"]


def test_schedule_and_page_furniture_do_not_contaminate_neighboring_clause():
    text = """ARTICLE 1 — INTERPRETATION
1.1 Definitions
Definitions apply to this Agreement.
SCHEDULE 1.5 TO EXHIBIT 1 — DEFINED TERMS
Page 5 10/7/26, 7:25 PM Master Services Agreement
Long Distance MSA -28- https://example.invalid/doc 33/71 Page 34
Defined Terms
Affiliate means an entity controlled by a party.
ARTICLE 2 — TERM
2.1 Term
The term begins on the Effective Date.
"""
    clauses = extract_clauses(text)
    first = next(clause for clause in clauses if clause["clause_number"] == "1.1")
    term = next(clause for clause in clauses if clause["clause_number"] == "2.1")
    assert first["article_title"] == "INTERPRETATION"
    assert "SCHEDULE" not in first["text"]
    assert term["article_title"] == "TERM"
    assert all("https://example.invalid" not in clause["text"] for clause in clauses)
