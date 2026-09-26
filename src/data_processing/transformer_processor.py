from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

import random
import json
import ast

CUAD_INPUT_PATH = Path("data/processed/cuad_clauses.json")
TRANSFORMER_INPUT_PATH = Path("data/processed/transformer_generated_dataset.json")
TRAIN_OUTPUT_PATH = Path("data/processed/transformer_train.json")
VAL_OUTPUT_PATH = Path("data/processed/transformer_validation.json")
TEST_OUTPUT_PATH = Path("data/processed/transformer_test.json")
LABEL_MAP_PATH = Path("data/processed/clause_label_mapping.json")

MODEL_NAME = "Saibo-creator/legal-roberta-base"
MAX_LENGTH = 256


def load_cuad_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def clean_clause_text(text):
    if text == "":
        return ""

    if text.startswith("[") and text.endswith("]"):
        parts = ast.literal_eval(text)

        if isinstance(parts, list):
            return " ".join(str(part) for part in parts)

    return text


def prepare_transformers_records(data):
    records = []
    for record in data:
        cleaned_text = clean_clause_text(record["text"])
        if record["clause_present"] is True and cleaned_text != "":
            new_record = {
                "contract_id": record["contract_id"],
                "text": cleaned_text,
                "clause_type": record["clause_type"],
            }
            records.append(new_record)
    return records


def save_json(data, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Saved: {file_path}")


def split_dataset_by_contract(records, train_ratio=0.8, val_ratio=0.1, seed=42):
    contracts = sorted(set(record["contract_id"] for record in records))

    random.seed(seed)
    random.shuffle(contracts)

    total_contracts = len(contracts)

    train_end = int(total_contracts * train_ratio)
    val_end = int(total_contracts * (train_ratio + val_ratio))

    train_contracts = set(contracts[:train_end])
    val_contracts = set(contracts[train_end:val_end])
    test_contracts = set(contracts[val_end:])

    train_records = []
    val_records = []
    test_records = []

    for record in records:
        contract_id = record["contract_id"]
        if contract_id in train_contracts:
            train_records.append(record)
        elif contract_id in val_contracts:
            val_records.append(record)
        elif contract_id in test_contracts:
            test_records.append(record)

    return (train_records, val_records, test_records)


def verify_contract_split(train_records, val_records, test_records):
    train_contracts = set(record["contract_id"] for record in train_records)
    val_contracts = set(record["contract_id"] for record in val_records)
    test_contracts = set(record["contract_id"] for record in test_records)

    train_val_overlap = train_contracts & val_contracts
    train_test_overlap = train_contracts & test_contracts
    val_test_overlap = val_contracts & test_contracts

    print("\n Contract split verification:")

    print(f"Train contracts: " f"{len(train_contracts)}")
    print(f"Validation contracts: " f"{len(val_contracts)}")
    print(f"Test contracts: " f"{len(test_contracts)}")
    print(f"Train ∩ Validation: " f"{len(train_val_overlap)}")
    print(f"Train ∩ Test: " f"{len(train_test_overlap)}")
    print(f"Validation ∩ Test: " f"{len(val_test_overlap)}")

    if (
        len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
    ):
        print("No contract leakage detected.")
    else:
        raise ValueError("Contract leakage detected!")


def create_label_mapping(records):
    clause_types = sorted(set(record["clause_type"] for record in records))

    label_to_id = {clause_type: index for index, clause_type in enumerate(clause_types)}

    id_to_label = {
        str(index): clause_type for clause_type, index in label_to_id.items()
    }

    mapping = {"label_to_id": label_to_id, "id_to_label": id_to_label}

    return mapping


def add_labels(records, label_to_id):
    labeled_records = []

    for record in records:

        new_record = {
            "contract_id": record["contract_id"],
            "text": record["text"],
            "clause_type": record["clause_type"],
            "label": label_to_id[record["clause_type"]],
        }

        labeled_records.append(new_record)

    return labeled_records


def tokenize_records(records, tokenizer):
    tokenized_records = []

    for record in records:

        encoded = tokenizer(
            record["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH
        )

        new_record = {
            "contract_id": record["contract_id"],
            "text": record["text"],
            "clause_type": record["clause_type"],
            "label": record["label"],
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

        tokenized_records.append(new_record)

    return tokenized_records


def main():
    # 1. Load CUAD
    data = load_cuad_data(CUAD_INPUT_PATH)
    print(f"Records loaded: {len(data)}")

    
    # 2. Prepare records
    records = prepare_transformers_records(data)
    print(f"Transformer records prepared: " f"{len(records)}")

    
    # 3. Verify clause categories
    clause_count = Counter(record["clause_type"] for record in records)
    print(f"\n {len(clause_count)} " "unique clause types found.")

    
    # 4. Save original prepared dataset
    save_json(records, TRANSFORMER_INPUT_PATH)

    
    # 5. Contract-level split
    train_records, val_records, test_records = split_dataset_by_contract(records)

    print(f"\n Train records: " f"{len(train_records)}")
    print(f"Validation records: " f"{len(val_records)}")
    print(f"Test records: " f"{len(test_records)}")

    
    # 6. Verify no contract leakage
    verify_contract_split(train_records, val_records, test_records)

    # 7. Create label mapping
    mapping = create_label_mapping(records)
    label_to_id = mapping["label_to_id"]
    save_json(mapping, LABEL_MAP_PATH)

    print(f"\n Number of labels: " f"{len(label_to_id)}")

    
    # 8. Add numerical labels
    train_records = add_labels(train_records, label_to_id)
    val_records = add_labels(val_records, label_to_id)
    test_records = add_labels(test_records, label_to_id)

    
    # 9. Load tokenizer
    print("\n Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizer loaded successfully.")
    print(f"Vocabulary size: " f"{tokenizer.vocab_size}")

    
    # 10. Tokenize datasets
    print("\n Tokenizing training records...")
    train_tokenized = tokenize_records(train_records, tokenizer)

    print("Training tokenization complete.")
    print("\n Tokenizing validation records...")

    val_tokenized = tokenize_records(val_records, tokenizer)

    print("Validation tokenization complete.")
    print("\n Tokenizing test records...")

    test_tokenized = tokenize_records(test_records, tokenizer)

    print("Test tokenization complete.")

    
    # 11. Save tokenized datasets
    save_json(train_tokenized, TRAIN_OUTPUT_PATH)
    save_json(val_tokenized, VAL_OUTPUT_PATH)
    save_json(test_tokenized, TEST_OUTPUT_PATH)

    
    # 12. Verify one tokenized record
    print("\n Tokenization verification:")

    sample = train_tokenized[0]

    print(f"Text: {sample['text']}")
    print(f"Clause type: " f"{sample['clause_type']}")
    print(f"Label: {sample['label']}")
    print(f"Input IDs length: " f"{len(sample['input_ids'])}")
    print(f"Attention mask length: " f"{len(sample['attention_mask'])}")


if __name__ == "__main__":
    main()
