from pathlib import Path
from collections import Counter

import random
import json
import ast

CUAD_INPUT_PATH = Path("data/processed/cuad_clauses.json")
TRANSFORMER_INPUT_PATH = Path("data/processed/transformer_generated_dataset.json")


def load_cuad_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return data


def clean_clause_text(text):
    if text == "":
        return text

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


def save_transformer_dataset(records, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)
    print(f"Transformer dataset saved at: {file_path}")


def split_dataset_by_contract(records, train_ratio=0.8, val_ratio=0.1, seed=42):
    contracts = list(set(record["contract_id"] for record in records))

    random.seed(seed)
    random.shuffle(contracts)

    total_contracts = len(contracts)

    train_end = int(len(contracts) * train_ratio)
    val_end = int(len(contracts) * (train_ratio + val_ratio))

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

    return train_records, val_records, test_records



def main():
    data = load_cuad_data(CUAD_INPUT_PATH)
    print(f"Records loaded: {len(data)}")

    records = prepare_transformers_records(data)
    print(f"Transformer records prepared: {len(records)}")
    print("\nSample record:")
    print(records[0])

    train_records, val_records, test_records = split_dataset_by_contract(records)
    print(f"\nTrain records: {len(train_records)}")
    print(f"Validation records: {len(val_records)}")
    print(f"Test records: {len(test_records)}")

    save_transformer_dataset(records, TRANSFORMER_INPUT_PATH)
    clause_count = Counter(record["clause_type"] for record in records)
    print(len(clause_count), "unique clause types found:")

    for clause_type, count in clause_count.items():
        print(f"{clause_type}: {count}")


if __name__ == "__main__":
    main()
