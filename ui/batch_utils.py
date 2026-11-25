import csv
from io import StringIO
from typing import Any, Dict, List


def parse_evaluation_csv(content: bytes) -> List[Dict[str, Any]]:
    """Parse CSV question/answer rows and optional reference columns."""
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV must be encoded as UTF-8.") from error

    reader = csv.DictReader(StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV must include a header row.")

    headers = {
        header.strip().lower(): header
        for header in reader.fieldnames
        if header is not None
    }
    answer_column = headers.get("ai_response") or headers.get("answer")
    missing_columns = []
    if "question" not in headers:
        missing_columns.append("question")
    if answer_column is None:
        missing_columns.append("ai_response (or answer)")
    if missing_columns:
        raise ValueError("CSV is missing required column(s): " + ", ".join(missing_columns))

    rows = []
    for row_number, row in enumerate(reader, start=2):
        question = (row.get(headers["question"]) or "").strip()
        answer = (row.get(answer_column) or "").strip()
        if not question or not answer:
            raise ValueError(
                f"CSV row {row_number} must include a non-empty question and answer."
            )
        rows.append({
            "question": question,
            "ai_response": answer,
            "reference_answer": _optional_value(row, headers.get("reference_answer")),
            "source_document": _optional_value(row, headers.get("source_document")),
        })

    if not rows:
        raise ValueError("CSV must contain at least one question/answer row.")
    return rows


def _optional_value(row, column):
    if column is None:
        return None
    value = row.get(column)
    return value.strip() if value and value.strip() else None
