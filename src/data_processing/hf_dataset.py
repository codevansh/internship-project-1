from pathlib import Path
from datasets import Dataset, DatasetDict

import json

DATA_DIR = Path("data/processed")

TRAIN_PATH = DATA_DIR / "transformer_train.json"
VAL_PATH = DATA_DIR / "transformer_validation.json"
TEST_PATH = DATA_DIR / "transformer_test.json"


def load_json(file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def main():

    print("Loading tokenized datasets...")

    train_data = load_json(TRAIN_PATH)
    val_data = load_json(VAL_PATH)
    test_data = load_json(TEST_PATH)

    print(f"Train records: {len(train_data)}")
    print(f"Validation records: {len(val_data)}")
    print(f"Test records: {len(test_data)}")

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)

    dataset = DatasetDict(
        {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
    )

    print("\n DatasetDict created successfully.")
    print(dataset)

    columns_to_remove = ["contract_id", "text", "clause_type"]

    dataset = dataset.remove_columns(columns_to_remove)

    print("\n Model-ready DatasetDict:")
    print(dataset)

    print("\n Final training columns:")
    print(dataset["train"].column_names)

    print("\n Sample training record:")
    print(dataset["train"][0])


if __name__ == "__main__":
    main()
