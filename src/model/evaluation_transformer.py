import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, Trainer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

MODEL_DIR = Path("models/legal_roberta_clause_classifier")
DATA_DIR = Path("data/processed")

TEST_PATH = DATA_DIR / "transformer_test.json"
LABEL_MAP_PATH = DATA_DIR / "clause_label_mapping.json"


def load_json(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_test_dataset():
    print("Loading test dataset...")

    test_data = load_json(TEST_PATH)
    test_dataset = Dataset.from_list(test_data)

    # Remove metadata that the model does not need
    columns_to_remove = [
        "contract_id",
        "text",
        "clause_type",
    ]

    test_dataset = test_dataset.remove_columns(columns_to_remove)
    print(f"Test records: {len(test_dataset)}")
    return test_dataset


def load_label_mapping():
    mapping = load_json(LABEL_MAP_PATH)

    label_to_id = mapping["label_to_id"]
    id_to_label = mapping["id_to_label"]

    return label_to_id, id_to_label


def compute_metrics(eval_prediction):
    predictions, labels = eval_prediction
    predicted_labels = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predicted_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def main():

    print("Loading label mapping...")

    label_to_id, id_to_label = load_label_mapping()
    print(f"Number of labels: {len(label_to_id)}")
    print("\nLoading test dataset...")

    test_dataset = load_test_dataset()
    print("\nLoading trained Legal-RoBERTa model...")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Trained model not found at: {MODEL_DIR}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    print("Model loaded successfully.")

    trainer = Trainer(
        model=model,
    )

    print("\nRunning predictions...")

    predictions = trainer.predict(test_dataset)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = np.array(predictions.label_ids)

    print("\n Evaluation Results")

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )

    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall   : {recall:.4f}")
    print(f"Macro F1       : {f1:.4f}")

    print("\n Classification Report")

    target_names = [id_to_label[str(i)] for i in range(len(id_to_label))]

    print(
        classification_report(
            true_labels,
            predicted_labels,
            labels=list(range(len(target_names))),
            target_names=target_names,
            zero_division=0,
        )
    )

    print("\n Confusion Matrix")

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
    )

    print(cm)


if __name__ == "__main__":
    main()
