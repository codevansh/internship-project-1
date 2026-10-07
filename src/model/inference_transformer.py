import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.model.classification_validation import validate_prediction

MODEL_NAME = "Saibo-creator/legal-roberta-base"
MODEL_DIR = "KAKAROT0304/legal-roberta-clause-classifier-day10"
LABEL_MAP_PATH = Path("data/processed/clause_label_mapping.json")
MAX_LENGTH = 256

def load_json(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_label_mapping():
    mapping = load_json(LABEL_MAP_PATH)
    label_to_id = mapping["label_to_id"]
    id_to_label = mapping["id_to_label"]
    return label_to_id, id_to_label


def load_model():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(
        "Tokenizer loaded successfully.\n"
        "Loading Day 10 Legal-RoBERTa model from Hugging Face"
    )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    model.eval()
    print("Day 10 model loaded successfully.")

    return tokenizer, model


def predict_clause(text, tokenizer, model, id_to_label, top_k=3):
    if not text or not text.strip():
        raise ValueError("Input clause text cannot be empty.")

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    # Disable gradient calculation for inference
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )

    # Convert logits into probabilities
    probabilities = torch.softmax(outputs.logits, dim=-1)

    # Get top predictions
    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=min(top_k, probabilities.shape[-1]),
        dim=-1,
    )

    top_probabilities = top_probabilities[0].tolist()
    top_indices = top_indices[0].tolist()

    predictions = []

    for probability, index in zip(top_probabilities, top_indices):
        label = id_to_label[str(index)]
        predictions.append(
            {
                "label": index,
                "clause_type": label,
                "confidence": round(float(probability), 4),
            }
        )

    result = {
        "input_text": text,
        "prediction": predictions[0],
        "top_predictions": predictions,
    }
    return result


def predict_contract_intelligence(text, tokenizer, model, id_to_label, nlp):
    """
    Run the complete contract intelligence pipeline.

    Legal-RoBERTa:
        Determines the clause type and classification confidence.

    spaCy NER:
        Extracts named entities from the clause.

    Risk engine:
        Extracts observable contractual indicators.
    """

    from src.risk.entity_extractor import extract_entities
    from src.risk.risk_engine import analyze_risk

    # Step 1: Legal-RoBERTa clause classification
    prediction_result = predict_clause(
        text,
        tokenizer,
        model,
        id_to_label,
        top_k=3,
    )

    raw_prediction = prediction_result["prediction"]
    prediction = validate_prediction(text, raw_prediction)

    # Step 2: spaCy entity extraction
    entities = extract_entities(
        nlp,
        text,
    )

    # Step 3: Contractual indicator extraction
    intelligence_result = analyze_risk(
        clause_type=prediction["clause_type"],
        classification_confidence=prediction["confidence"],
        clause_text=text,
        entities=entities,
    )

    # Preserve top model predictions as additional information
    intelligence_result["top_predictions"] = prediction_result["top_predictions"]
    intelligence_result["classification_validation"] = prediction.get("validation", {})
    if prediction.get("validation", {}).get("status") == "unsupported_label":
        intelligence_result["risk_assessment"]["requires_human_review"] = True
        intelligence_result["risk_assessment"]["reasons"].append(
            "Predicted clause category was not supported by the clause text; classification needs review."
        )
        intelligence_result["predicted_clause_type"] = raw_prediction["clause_type"]

    return intelligence_result


def main():

    print("LEGAL CONTRACT CLAUSE INFERENCE")
    print("\n Loading label mapping...")

    _, id_to_label = load_label_mapping()
    print(f"Number of clause classes: {len(id_to_label)}")

    tokenizer, model = load_model()

    # Test clauses
    test_clauses = [
        (
            "The agreement shall be governed by and construed "
            "in accordance with the laws of the State of New York."
        ),
        (
            "Neither party may assign or transfer this agreement "
            "without the prior written consent of the other party."
        ),
        (
            "Either party may terminate this agreement upon "
            "thirty days written notice to the other party."
        ),
    ]
    print("Running test predicitions")

    for i, clause in enumerate(test_clauses, start=1):

        print(f"\n Test Clause {i}")
        print(f"Text:\n {clause}")

        result = predict_clause(
            clause,
            tokenizer,
            model,
            id_to_label,
            top_k=3,
        )

        prediction = result["prediction"]
        print("\nPrediction:")
        print(f"Clause Type : {prediction['clause_type']}")
        print(f"Label       : {prediction['label']}")
        print(f"Confidence  : {prediction['confidence']:.4f}")

        print("\n Top 3 Predictions:")
        for rank, item in enumerate(result["top_predictions"], start=1):
            print(f"{rank}. " f"{item['clause_type']} " f"({item['confidence']:.4f})")

    print("Inference completed")


if __name__ == "__main__":
    main()
