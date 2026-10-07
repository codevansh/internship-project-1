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
