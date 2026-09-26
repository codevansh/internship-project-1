import json
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "Saibo-creator/legal-roberta-base"
DATA_DIR = Path("data/processed")

TRAIN_PATH = DATA_DIR / "transformer_train.json"
VAL_PATH = DATA_DIR / "transformer_validation.json"
TEST_PATH = DATA_DIR / "transformer_test.json"
LABEL_MAP_PATH = DATA_DIR / "clause_label_mapping.json"

OUTPUT_DIR = Path("models/legal_roberta_clause_classifier")


def load_json(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset():
    print("Loading tokenized datasets...")

    train_data = load_json(TRAIN_PATH)
    val_data = load_json(VAL_PATH)
    test_data = load_json(TEST_PATH)

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)

    dataset = DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    )

    # Remove metadata that the model should not receive
    columns_to_remove = ["contract_id", "text", "clause_type"]
    dataset = dataset.remove_columns(columns_to_remove)
    return dataset


def load_label_mapping():
    mapping = load_json(LABEL_MAP_PATH)
    label_to_id = mapping["label_to_id"]
    id_to_label = mapping["id_to_label"]

    return label_to_id, id_to_label


def main():
    # 1. Load dataset
    dataset = load_dataset()
    print("\n Dataset loaded:")
    print(dataset)

    # 2. Load labels
    label_to_id, id_to_label = load_label_mapping()
    num_labels = len(label_to_id)
    print(f"\n Number of labels: {num_labels}")

    # 3. Load model
    print("\n Loading Legal-RoBERTa...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, label2id=label_to_id, id2label=id_to_label
    )

    print("Model loaded successfully.")

    # 4. Small sanity-check configuration
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        report_to="none",
        fp16=False,
    )

    # 5. Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )

    print("\n Starting short transformer training run...")

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    print("\n short transformer training run completed successfully.")


if __name__ == "__main__":
    main()
