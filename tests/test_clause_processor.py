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
