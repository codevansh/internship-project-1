import re
from pathlib import Path
from typing import Dict, List

OCR_INPUT_PATH = Path("data/processed/ocr_sample.txt")


ARTICLE_PATTERN = re.compile(
    r"^\s*ARTICLE\s*(\d+)\s*$",
    re.IGNORECASE,
)


CLAUSE_PATTERN = re.compile(
    r"^\s*(\d+\.\d+)\s+(.+?)\s*$",
)


def load_ocr_text(file_path: Path = OCR_INPUT_PATH) -> str:
    """Load OCR text from the processed OCR file."""

    if not file_path.exists():
        raise FileNotFoundError(f"OCR text not found at: {file_path}")
    return file_path.read_text(encoding="utf-8")


def is_article_heading(line: str) -> bool:
    """Check whether a line is an ARTICLE heading."""
    return bool(ARTICLE_PATTERN.match(line.strip()))


def parse_clause_heading(line: str):
    """Parse a numbered clause heading.
    Handles examples such as:
        1.3 Delivery; Acceptance.
    and:
        1.2 Title and Risk of Loss. With respect to the Products...
    Returns:
        {
            "clause_number": "...",
            "title": "...",
            "initial_text": "..."
        }
    Returns None if the line is not a clause heading."""

    match = CLAUSE_PATTERN.match(line.strip())
    if not match:
        return None

    clause_number = match.group(1)
    remainder = match.group(2).strip()

    # Try to separate the clause title from the body text.
    # Example:
    # "Title and Risk of Loss. With respect to the Products..."
    # becomes:
    # title:
    # "Title and Risk of Loss."
    # initial_text:
    # "With respect to the Products..."
    title_match = re.match(
        r"^(.*?[.!?])\s+(?=[A-Z(\[])",
        remainder,
    )

    if title_match:
        title = title_match.group(1).strip()
        remaining_text = remainder[title_match.end() :].strip()
    else:
        title = remainder
        remaining_text = ""
    return {
        "clause_number": clause_number,
        "title": title,
        "initial_text": remaining_text,
    }


def extract_clauses(text: str) -> List[Dict]:
    """Extract individual clauses from OCR text.
    Each clause contains:
        article
        clause_number
        title
        text"""

    lines = text.splitlines()
    clauses = []

    current_article = None
    current_clause = None
    current_text = []

    def save_current_clause():
        """Save the currently collected clause."""

        if current_clause is None:
            return

        clause_text = " ".join(
            line.strip() for line in current_text if line.strip()
        ).strip()

        if not clause_text:
            return

        clauses.append(
            {
                "article": current_clause["article"],
                "clause_number": current_clause["clause_number"],
                "title": current_clause["title"],
                "text": clause_text,
            }
        )

    for line in lines:
        stripped = line.strip()

        # Ignore empty lines.
        if not stripped:
            continue

        # ARTICLE HEADING

        if is_article_heading(stripped):
            save_current_clause()

            current_clause = None
            current_text = []

            article_match = ARTICLE_PATTERN.match(stripped)
            if article_match:
                current_article = article_match.group(1)
            continue

        # CLAUSE HEADING
        clause_heading = parse_clause_heading(stripped)
        if clause_heading is not None:

            # Save the previous clause before starting a new one.
            save_current_clause()

            clause_number = clause_heading["clause_number"]
            clause_article = clause_number.split(".")[0]

            current_clause = {
                "article": clause_article,
                "clause_number": clause_number,
                "title": clause_heading["title"],
            }

            current_text = []

            # If the clause heading and body text
            # are on the same OCR line, preserve
            # the body text.
            if clause_heading["initial_text"]:
                current_text.append(clause_heading["initial_text"])

            # This line has been completely handled.
            continue

        # NORMAL CLAUSE BODY TEXT
        if current_clause is not None:
            current_text.append(stripped)

    # Save the final clause.
    save_current_clause()
    return clauses


def main():
    """Run clause extraction on the OCR contract."""

    print("Contract clause extraction")

    # Load OCR text.
    text = load_ocr_text()

    print(f"OCR file loaded: {OCR_INPUT_PATH}")

    # Extract clauses.
    clauses = extract_clauses(text)

    print(f"\nClauses extracted: {len(clauses)}")

    # Display first 10 clauses for verification.
    for clause in clauses[:10]:
        print("\n")
        print(f"Article: {clause['article']}")
        print(f"Clause: {clause['clause_number']}")
        print(f"Title: {clause['title']}")
        print(f"Text: {clause['text'][:300]}...")
    print("Clause extraction completed successfully.")


if __name__ == "__main__":
    main()
