import re
from pathlib import Path
from typing import Dict, List

OCR_INPUT_PATH = Path("data/processed/ocr_sample.txt")

ARTICLE_PATTERN = re.compile(
    r"^\s*ARTICLE\s+([\w.-]+)(?:(?:\s*[-—:]\s*|\s+)(.+))?\s*$", re.IGNORECASE
)
SECTION_PATTERN = re.compile(
    r"^\s*SECTION\s+([\w.-]+)(?:(?:\s*[-—:]\s*|\s+)(.+))?\s*$", re.IGNORECASE
)
NUMBERED_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)*(?:\.)?)\s+(.+?)\s*$")
ROMAN_ARTICLE_PATTERN = re.compile(r"^\s*ARTICLE\s+([IVXLCDM]+)\s*$", re.IGNORECASE)
RECITAL_PATTERN = re.compile(
    r"^\s*(RECITALS?|WHEREAS\b.*|NOW,?\s+THEREFORE\b.*)\s*$", re.IGNORECASE
)
PAGE_MARKER = re.compile(r"^(?:page\s+)?\d+(?:\s+of\s+\d+)?$", re.IGNORECASE)
PAGE_HEADER = re.compile(
    r"^page\s+\d+\s+.*(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}:\d{2}\s*[ap]m).*",
    re.IGNORECASE,
)
PAGE_FOOTER = re.compile(r"(?:https?://|\b\d+/\d+\s+page\s+\d+)", re.IGNORECASE)
TOC_LINE = re.compile(r"(?:\.{2,}|_{2,})\s*\d{1,4}\s*$")
NUMBERED_TOC_LINE = re.compile(
    r"^(?:(?:article|section)\s+)?(?:\d+(?:\.\d+)*|[ivxlcdm]+)\s+.{2,}\s+\d{1,3}\s*$",
    re.IGNORECASE,
)
ADDRESS_LINE = re.compile(
    r"\b(?:street|st\.?|avenue|ave\.?|boulevard|blvd\.?|road|rd\.?|drive|dr\.?|lane|ln\.?|suite|floor|highway|hwy\.?|p\.?o\.?\s*box)\b|\b\d{5}(?:-\d{4})?\b",
    re.IGNORECASE,
)
ARTIFACT_LINE = re.compile(
    r"^(?:legal_\d+\s*:.*|\*{3,}|\[?(?:\*{3,}|redacted|page\s+\d+|confidential)\]?|(?:exhibit|schedule|appendix)\s+[a-z0-9.-]+)$",
    re.IGNORECASE,
)
REDACTION_NOTICE = re.compile(
    r"(?:information has been excluded from this exhibit|indicates that information has been redacted|competitive harm to the registrant|^been redacted\.?$)",
    re.IGNORECASE,
)
SCHEDULE_HEADING = re.compile(
    r"^\s*(?:SCHEDULE|EXHIBIT|APPENDIX|ATTACHMENT)\s+[A-Z0-9.-]+(?:\s+.*)?$",
    re.IGNORECASE,
)
ADDENDUM_CAPTION = re.compile(
    r"^\s*SERVICE ADDENDUM\s+(?:ONE|TWO|\d+)\b.*$", re.IGNORECASE
)


def load_ocr_text(file_path: Path = OCR_INPUT_PATH) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"OCR text not found at: {file_path}")
    return file_path.read_text(encoding="utf-8")


def is_article_heading(line: str) -> bool:
    return bool(ARTICLE_PATTERN.match(line.strip()))


def _split_title(remainder: str):
    title_match = re.match(r"^(.*?[.!?])\s+(?=[A-Z(\[])", remainder)
    if title_match:
        return title_match.group(1).strip(), remainder[title_match.end() :].strip()
    return remainder.strip().rstrip("."), ""


def parse_clause_heading(line: str):
    match = NUMBERED_PATTERN.match(line.strip())
    if not match:
        return None
    number = match.group(1).rstrip(".")
    title, initial_text = _split_title(match.group(2).strip())
    return {"clause_number": number, "title": title, "initial_text": initial_text}


def _new_clause(
    number,
    title,
    article,
    section=None,
    article_title=None,
    section_title=None,
    schedule=None,
):
    return {
        "article": article,
        "article_title": article_title,
        "section": section,
        "section_title": section_title,
        "schedule": schedule,
        "clause_number": number,
        "title": title,
    }


def extract_clauses(text: str) -> List[Dict]:
    """Extract clauses from common legal headings, with a paragraph fallback.

    ARTICLE/SECTION headings are retained as metadata, but numbered clauses and
    unnumbered paragraphs are independently recognized so either can stand alone.
    """
    clauses = []
    raw_lines = [line.strip() for line in text.splitlines()]
    # Repeated all-caps titles and address/header lines are usually page furniture.
    normalized_counts = {}
    for line in raw_lines:
        if line:
            key = re.sub(r"\s+", " ", line).casefold()
            normalized_counts[key] = normalized_counts.get(key, 0) + 1

    def is_artifact(line: str) -> bool:
        if (
            not line
            or PAGE_MARKER.fullmatch(line)
            or PAGE_HEADER.fullmatch(line)
            or ARTIFACT_LINE.fullmatch(line)
            or REDACTION_NOTICE.search(line)
            or (PAGE_FOOTER.search(line) and re.search(r"\bpage\s+\d+", line, re.I))
        ):
            return True
        if (
            (TOC_LINE.search(line) or NUMBERED_TOC_LINE.fullmatch(line))
            and re.search(r"\d\s*$", line)
            and not re.search(r"[$%]", line)
        ):
            return True
        if ADDRESS_LINE.search(line):
            # Filter address blocks; retain contract sentences that happen to cite an address.
            has_obligation = re.search(
                r"\b(?:shall|must|may|agrees?|notify|provide|deliver|send)\b",
                line,
                re.I,
            )
            looks_like_address = re.match(r"^\d+\s+\w+", line) or line.count(",") >= 1
            if not has_obligation and (
                looks_like_address
                or not re.search(r"\b(?:is|are|was|were|located|at|to)\b", line, re.I)
            ):
                return True
        key = re.sub(r"\s+", " ", line).casefold()
        if normalized_counts.get(key, 0) > 1 and len(line) < 120 and line.isupper():
            return True
        # OCR often leaves isolated scraps such as "dee" between page furniture.
        # Punctuated short provisions (for example, "Pay.") remain eligible.
        if re.fullmatch(r"[A-Za-z]{1,3}", line):
            return True
        return False

    current_article = None
    current_article_title = None
    current_section = None
    current_section_title = None
    current_schedule = None
    current_clause = None
    current_text = []
    unnumbered_index = 0

    def save():
        if current_clause is None:
            return
        body = " ".join(part.strip() for part in current_text if part.strip()).strip()
        if body:
            clauses.append({**current_clause, "text": body})

    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue
        if is_artifact(line):
            continue
        if SCHEDULE_HEADING.fullmatch(line) or ADDENDUM_CAPTION.fullmatch(line):
            save()
            current_clause, current_text = None, []
            current_schedule = line
            continue
        article_match = ARTICLE_PATTERN.match(line)
        section_match = SECTION_PATTERN.match(line)
        if article_match or section_match:
            save()
            current_clause, current_text = None, []
            if article_match:
                current_article = article_match.group(1)
                current_article_title = (
                    re.sub(r"^[^\w]+", "", article_match.group(2) or "") or None
                )
                current_section = None
                current_section_title = None
                if article_match.group(2):
                    # Article captions are structure, not a standalone clause.
                    continue
            if section_match:
                current_section = section_match.group(1)
                current_section_title = (
                    re.sub(r"^[^\w]+", "", section_match.group(2) or "") or None
                )
                if section_match.group(2):
                    continue
            continue

        # Standalone article captions commonly follow "ARTICLE 1" on the next line.
        if (
            current_clause is None
            and current_article
            and not current_article_title
            and line.isupper()
        ):
            current_article_title = line
        if (
            current_clause is None
            and line.upper() != "RECITALS"
            and re.fullmatch(r"[A-Z][A-Z0-9 ,&/()'’-]{2,100}", line)
        ):
            continue

        heading = parse_clause_heading(line)
        if heading:
            save()
            number = heading["clause_number"]
            derived_article = number.split(".")[0]
            article = current_article
            if not article or str(article).isdigit() and article != derived_article:
                article = derived_article
            current_clause = _new_clause(
                number,
                heading["title"],
                article,
                current_section,
                current_article_title,
                current_section_title,
                current_schedule,
            )
            current_text = [heading["initial_text"]] if heading["initial_text"] else []
            continue

        recital = RECITAL_PATTERN.match(line)
        if recital and (
            line.upper() == "RECITALS"
            or line.upper().startswith("WHEREAS")
            or line.upper().startswith("NOW")
        ):
            save()
            unnumbered_index += 1
            current_clause = _new_clause(
                f"R{unnumbered_index}",
                line.rstrip(".;"),
                current_article,
                current_section,
                current_article_title,
                current_section_title,
                current_schedule,
            )
            current_text = [line]
            continue

        if current_clause is None:
            # Keep prose introductions, while filtering document furniture above.
            unnumbered_index += 1
            current_clause = _new_clause(
                f"U{unnumbered_index}",
                "Unnumbered provision",
                current_article,
                current_section,
                current_article_title,
                current_section_title,
                current_schedule,
            )
            current_text = [line]
        else:
            current_text.append(line)

    save()
    return clauses


def main():
    text = load_ocr_text()
    clauses = extract_clauses(text)
    print(f"Clauses extracted: {len(clauses)}")
    for clause in clauses[:10]:
        print(f"{clause['clause_number']} {clause['title']}: {clause['text'][:300]}")


if __name__ == "__main__":
    main()
