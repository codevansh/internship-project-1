import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

# Project paths

RAW_DATA_PATH = Path("data/raw/master_clauses.csv")
PROCESSED_DATA_PATH = Path("data/processed/cuad_clauses.json")
TOKENIZED_DATA_PATH = Path("data/processed/cuad_tokenized_sample.json")

TOKENIZER_NAME = "bert-base-uncased"


# loading and printing the rows and columns of the dataset using pandas.
def load_cuad():
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"CUAD dataset not found at: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Loaded CUAD dataset: {df.shape[0]} contracts")
    print(f"Number of columns: {df.shape[1]}")

    return df


def get_clause_columns(df):

    answer_columns = {
        column
        for column in df.columns
        if column.endswith("-Answer") or column.endswith("- Answer")
    }  # finds column to column-answer pairs

    clause_columns = []

    for column in df.columns:
        if column == "Filename":
            continue

        if column in answer_columns:
            continue

        answer_column = None

        possible_names = [
            f"{column}-Answer",
            f"{column}- Answer",
        ]

        for possible_name in possible_names:
            if possible_name in df.columns:
                answer_column = possible_name
                break

        if answer_column:
            clause_columns.append((column, answer_column))

    return clause_columns


def normalize_cuad(df):

    clause_pairs = get_clause_columns(df)
    records = []

    for _, row in df.iterrows():
        contract_id = row["Filename"]

        for clause_name, answer_column in clause_pairs:

            clause_value = row[clause_name]
            answer_value = row[answer_column]

            if pd.isna(clause_value):
                clause_value = ""

            if pd.isna(answer_value):
                answer_value = ""

            clause_present = str(clause_value).strip() != "[]"
            clause_value = str(clause_value).strip()
            answer_value = str(answer_value).strip()

            record = {
                "contract_id": contract_id,
                "clause_type": clause_name,
                "clause_present": clause_present,
                "answer": answer_value,
            }

            records.append(record)
    # print(clause_name, clause_value, answer_value)
    return records


def save_json(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)

    print(f"Saved {len(records)} records to:")
    print(output_path)


def tokenize_sample(records, sample_size=20):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    sample = records[:sample_size]
    tokenized_records = []

    for record in sample:
        answer = record["answer"]

        encoded = tokenizer(
            answer,
            truncation=True,
            padding=False,
            max_length=512,
        )

        tokenized_record = {
            **record,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

        tokenized_records.append(tokenized_record)

    return tokenized_records


def main():
    df = load_cuad()

    clause_pairs = get_clause_columns(df)
    print(f"\n Identified clause categories: {len(clause_pairs)}")

    records = normalize_cuad(df)

    inconsistent_count = check_record_consistency(records)
    print(f" \n  Inconsisten Records:{inconsistent_count}")

    a, b, c, d = validate_records(records)
    print(f"Missing contract_id: {a}")
    print(f"Missing clause_type: {b}")
    print(f"Missing clause_present: {c}")
    print(f"Missing answer: {d}")

    # clause_distribution = analyze_records_count(records)
    # print(clause_distribution)

    clause_answer_distribution = analyze_answer_count(records)
    print(f" \n  Unique Answers count: {len(clause_answer_distribution)}")

    top_answers = sorted(
        clause_answer_distribution.items(), key=lambda x: x[1], reverse=True
    )[:10]
    for answer, count in top_answers:
        print(answer, count)

    print(f" \n Generated normalized records: {len(records)}")
    save_json(records, PROCESSED_DATA_PATH)

    tokenized_records = tokenize_sample(records)
    save_json(tokenized_records, TOKENIZED_DATA_PATH)
    print("\n CUAD processing completed successfully.")

    summary = dataset_summary(records)
    print(summary)


def validate_records(records):
    not_id_count = 0
    not_clause_type = 0
    not_clause_present = 0
    not_answer = 0

    for every_record in records:
        if not "contract_id" in every_record:
            not_id_count = not_id_count + 1
        if not "clause_type" in every_record:
            not_clause_type = not_clause_type + 1
        if not "clause_present" in every_record:
            not_clause_present = not_clause_present + 1
        if not "answer" in every_record:
            not_answer = not_answer + 1
    return (
        not_id_count,
        not_clause_type,
        not_clause_present,
        not_answer,
    )


# Day-3
def analyze_records_count(records):
    record_type = {}

    for i in records:
        clause = i["clause_type"]
        clause_present = i["clause_present"]

        if clause not in record_type:
            record_type[clause] = {"pos_count": 0, "neg_count": 0}

        if clause_present:
            record_type[clause]["pos_count"] += 1
        else:
            record_type[clause]["neg_count"] += 1

    return record_type


def analyze_answer_count(records):
    answers_count = {}

    for i in records:
        answer = i["answer"]
        if answer in answers_count:
            answers_count[answer] += 1
        else:
            answers_count[answer] = 1

    return answers_count


def check_record_consistency(records):
    inconsistent_count = 0
    for i in records:
        contract_id = i["contract_id"]
        if not contract_id:
            inconsistent_count += 1

        clause_type = i["clause_type"]
        if clause_type == "":
            inconsistent_count += 1

        clause_present = i["clause_present"]
        if not isinstance(clause_present, bool):
            inconsistent_count += 1

        answer = i["answer"]
        if not isinstance(answer, str):
            inconsistent_count += 1

    return inconsistent_count


def dataset_summary(records):
    total_records = len(records)

    contract_ids = set()
    clause_types = set()
    answers = set()

    for i in records:
        contract_ids.add(i["contract_id"])
        clause_types.add(i["clause_type"])
        answers.add(i["answer"])

    return {
        "total_records": total_records,
        "unique_contracts": len(contract_ids),
        "unique_clause_types": len(clause_types),
        "unique_answers": len(answers),
    }


if __name__ == "__main__":
    main()
