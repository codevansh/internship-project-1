import json
from pathlib import Path

from src.model.inference_transformer import (
    load_label_mapping,
    load_model,
    predict_contract_intelligence,
)
from src.risk.entity_extractor import load_ner_model

OUTPUT_PATH = Path("data/processed/contract_intelligence_output.json")


def load_contract_intelligence_models():
    """
    Load all models required for contract intelligence.

    Returns:
        tokenizer: Legal-RoBERTa tokenizer
        model: trained Day-10 Legal-RoBERTa classifier
        id_to_label: clause label mapping
        nlp: spaCy NER model
    """

    _, id_to_label = load_label_mapping()
    tokenizer, model = load_model()
    nlp = load_ner_model()

    return tokenizer, model, id_to_label, nlp


def analyze_clause(clause_text, tokenizer, model, id_to_label, nlp):
    """ Analyze a single contract clause.

    Pipeline:
        Legal-RoBERTa classification + spaCy NER + contractual indicator extraction """

    if not clause_text or not clause_text.strip():
        raise ValueError("Clause text cannot be empty.")

    return predict_contract_intelligence(
        clause_text,
        tokenizer,
        model,
        id_to_label,
        nlp,
    )


def save_results(results, output_path=OUTPUT_PATH):
    """ Save contract intelligence results as JSON. """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path,"w",encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    return output_path
