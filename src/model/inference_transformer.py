import json
import logging
import os
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import huggingface_hub
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.model.classification_validation import validate_prediction

MODEL_NAME = "Saibo-creator/legal-roberta-base"
MODEL_DIR = "KAKAROT0304/legal-roberta-clause-classifier-day10"
LABEL_MAP_PATH = Path("data/processed/clause_label_mapping.json")
MAX_LENGTH = 256
logger = logging.getLogger(__name__)


def _safe_hf_endpoint():
    """Return only the active Hub endpoint's scheme and host for diagnostics."""
    try:
        from huggingface_hub.constants import ENDPOINT

        parsed = urlsplit(ENDPOINT)
        if parsed.scheme and parsed.hostname:
            host = parsed.hostname
            if parsed.port:
                host = f"{host}:{parsed.port}"
            return f"{parsed.scheme}://{host}"
    except (ImportError, ValueError):
        pass
    return "unknown"


def _redact_url(match):
    """Keep a request URL useful for diagnosis while dropping credentials/query."""
    try:
        parsed = urlsplit(match.group(0).rstrip(".,);]"))
        if not parsed.scheme or not parsed.hostname:
            return "<redacted-url>"
        host = parsed.hostname
        if parsed.port:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "<redacted>", ""))
    except ValueError:
        return "<redacted-url>"


def _safe_error_message(error):
    """Redact credentials and keep the underlying exception detail readable."""
    message = str(error)
    message = re.sub(r"https?://[^\s<>\"']+", _redact_url, message)
    message = re.sub(r"(?i)\bBearer\s+\S+", "Bearer <redacted>", message)
    message = re.sub(r"(?i)\bBasic\s+\S+", "Basic <redacted>", message)
    message = re.sub(r"\bhf_[A-Za-z0-9_-]{10,}\b", "<redacted HF token>", message)
    message = re.sub(
        r"(?i)\b(authorization|proxy-authorization|access[_-]?token|token|api[_-]?key|password|client[_-]?secret)\b\s*[:=]\s*[^\s,;]+",
        r"\1=<redacted>",
        message,
    )
    return message.replace("\r", "\\r").replace("\n", "\\n")


def _exception_chain(error):
    chain = []
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def _log_hf_load_failure(stage, repo_id, error):
    chain = _exception_chain(error)
    response = next(
        (getattr(item, "response", None) for item in chain if getattr(item, "response", None) is not None),
        None,
    )
    status_code = getattr(response, "status_code", None)
    response_url = getattr(response, "url", None)
    safe_response_url = _safe_error_message(response_url) if response_url else "unavailable"
    cause_chain = " <- ".join(
        f"{type(item).__name__}: {_safe_error_message(item)}" for item in chain
    )
    offline = any(
        os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )
    proxy_configured = any(
        os.getenv(name)
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
    )
    logger.error(
        "Hugging Face %s load failed: repo=%s endpoint=%s response_url=%s "
        "http_status=%s exception_type=%s underlying_exception_type=%s "
        "transformers=%s huggingface_hub=%s offline=%s proxy_configured=%s "
        "cause_chain=%s",
        stage,
        repo_id,
        _safe_hf_endpoint(),
        safe_response_url,
        status_code if status_code is not None else "unavailable",
        type(error).__name__,
        type(chain[-1]).__name__ if chain else type(error).__name__,
        transformers.__version__,
        huggingface_hub.__version__,
        offline,
        bool(proxy_configured),
        cause_chain,
    )

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

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as error:
        _log_hf_load_failure("tokenizer", MODEL_NAME, error)
        raise RuntimeError(
            f"Could not load the Legal-RoBERTa tokenizer '{MODEL_NAME}'. "
            "Check that the Hugging Face repository is available and that "
            "the service has network access and permission to download it."
        ) from error

    print(
        "Tokenizer loaded successfully.\n"
        "Loading Day 10 Legal-RoBERTa model from Hugging Face"
    )

    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    except Exception as error:
        _log_hf_load_failure("classifier", MODEL_DIR, error)
        raise RuntimeError(
            f"Could not load the trained clause classifier '{MODEL_DIR}'. "
            "Verify that this Hugging Face model ID is correct, exists, and "
            "is accessible to the service (including HF_TOKEN if it is private). "
            "The training script saves its checkpoint to "
            "'models/legal_roberta_clause_classifier'; deploy a compatible "
            "trained checkpoint explicitly if using that local output. "
            "No untrained or substitute model will be used."
        ) from error

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

    # Inference mode avoids autograd bookkeeping and version tracking.
    with torch.inference_mode():
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

    top_probabilities_tensor = top_probabilities
    top_indices_tensor = top_indices
    top_probabilities = top_probabilities_tensor[0].tolist()
    top_indices = top_indices_tensor[0].tolist()

    # Do not keep per-clause tensors alive while the rest of the clause is
    # processed (including spaCy entity extraction and report construction).
    del encoded, outputs, probabilities, top_probabilities_tensor, top_indices_tensor

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
