import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

# Project paths

RAW_DATA_PATH = Path("data/raw/master_clauses.csv")
PROCESSED_DATA_PATH = Path("data/processed/cuad_clauses.json")
TOKENIZED_DATA_PATH = Path("data/processed/cuad_tokenized_sample.json")
ML_DATA_PATH = Path("data/processed/cuad_ml_records.json")

TOKENIZER_NAME = "bert-base-uncased"

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


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

            clause_value = str(clause_value).strip()
            answer_value = str(answer_value).strip()

            if clause_value == "[]":
                clause_value = ""
            clause_present = clause_value != ""

            record = {
                "contract_id": contract_id,
                "clause_type": clause_name,
                "text": clause_value,
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
        text = record["text"]

        encoded = tokenizer(
            text,
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
    labeled_records = create_labels(records)
    ml_records = prepare_ml_records(labeled_records)
    print(ml_records[0])
    print(ml_records[8])

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

    print(f"\nGenerated ML records: {len(ml_records)}")
    save_json(ml_records, ML_DATA_PATH)

    tokenized_records = tokenize_sample(records)
    save_json(tokenized_records, TOKENIZED_DATA_PATH)
    print("\n CUAD processing completed successfully.")

    summary = dataset_summary(records)
    print(summary)

    train_records, test_records = splitting(ml_records)
    print(f"Train records: {len(train_records)}")
    print(f"Test records: {len(test_records)}")

    overlapping_contracts = check_split(train_records, test_records)
    print(f"\n OVerlapping contracts: {overlapping_contracts}")

    X_train, X_test, y_train, y_test = prepare_ml_data(train_records, test_records)

    # print(f"X_train: {len(X_train)}")
    # print(f"X_test: {len(X_test)}")
    # print(f"y_train: {len(y_train)}")
    # print(f"y_test: {len(y_test)}")

    X_train_tfidf, X_test_tfidf, vectorizer = create_tfidf_features(X_train, X_test)

    model = train_model(X_train_tfidf, y_train)
    # print("\nLogistic Regression model trained successfully.")

    predictions = predict_model_values(model, X_test_tfidf)
    # print(f"\n Predicting values: {len(predictions)}")

    class_rep = classification_report(y_test, predictions)
    accuracy, precision, recall, f1, cm, class_rep = evaluate_model(y_test, predictions)

    errors = analyze_errors(test_records, predictions)

    # print(f"\nAccuracy: {accuracy:.4f}")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    # print(f"Confusion Matrix:\n{cm}")
    # print(f"\n Classification Report:\n{class_rep}")
    print(f"\n False Positives: {len(errors[0])}")
    print(f"\n False Negatives: {len(errors[1])}")


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


def create_labels(records):
    labeled_records = []

    for i in records:
        copy_record = i.copy()

        if copy_record["clause_present"]:
            copy_record["label"] = 1
        else:
            copy_record["label"] = 0
        labeled_records.append(copy_record)

    return labeled_records


def prepare_ml_records(records):
    ml_records = []
    ml_record = {}
    for i in records:
        ml_record = dict(i)
        ml_record.pop("answer")
        ml_record.pop("clause_present")

        ml_records.append(ml_record)

    return ml_records


# day-4 splitting records.
def splitting(records):
    contract_ids = sorted({record["contract_id"] for record in records})

    train_ids, test_ids = train_test_split(contract_ids, test_size=0.2, random_state=42)

    train_ids = set(train_ids)
    test_ids = set(test_ids)

    train_records = []
    test_records = []

    for record in records:
        if record["contract_id"] in train_ids:
            train_records.append(record)
        else:
            test_records.append(record)

    return train_records, test_records


def check_split(train_records, test_records):
    train_ids = set()
    test_ids = set()

    for i in train_records:
        train_ids.add(i["contract_id"])

    for i in test_records:
        test_ids.add(i["contract_id"])

    overlap = train_ids.intersection(test_ids)
    return overlap


def prepare_ml_data(train_records, test_records):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in train_records:
        # X_train.append(i["clause_type"] + " " + i["text"])
        X_train.append(i["text"])
        y_train.append(i["label"])

    for i in test_records:
        # X_test.append(i["clause_type"] + " " + i["text"])
        X_test.append(i["text"])
        y_test.append(i["label"])

    return X_train, X_test, y_train, y_test


def create_tfidf_features(X_train, X_test):
    vectorizer = TfidfVectorizer()

    X_train_tfidf = vectorizer.fit_transform(
        X_train
    )  # we do fit_transform on X_train data so that the model learns from it and then converts it into numbers.
    X_test_tfidf = vectorizer.transform(
        X_test
    )  # we dont do fit transfomr on X_Test data becuase it is test data.

    return X_train_tfidf, X_test_tfidf, vectorizer


def train_model(X_train_tfidf, y_train):
    model = LogisticRegression(class_weight="balanced")
    model.fit(X_train_tfidf, y_train)
    return model


def predict_model_values(model, X_test_tfidf):
    probabilities = model.predict_proba(X_test_tfidf)[:, 1]
    threshold = 0.45
    predictions = (probabilities >= threshold).astype(int)
    return predictions


def evaluate_model(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    class_rep = classification_report(y_test, predictions)
    return (accuracy, precision, recall, f1, cm, class_rep)


def analyze_errors(test_records, predictions):
    fp = []
    fn = []
    for i, record in enumerate(test_records):
        if record["label"] == 0 and predictions[i] == 1:
            fp.append(record)
        elif record["label"] == 1 and predictions[i] == 0:
            fn.append(record)
    return fp, fn


if __name__ == "__main__":
    main()
