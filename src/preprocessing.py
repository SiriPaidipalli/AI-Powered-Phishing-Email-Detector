"""Shared preprocessing for canonical subject/body/label email records.

Dataset-specific importers should map their source format to these three fields.
This module deliberately does not guess mappings for unknown datasets.
"""

import csv
import hashlib
import os
import re
import tempfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterator, Mapping, TextIO, Tuple


URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b", re.IGNORECASE)
REQUIRED_COLUMNS = ("subject", "body", "label")
OUTPUT_COLUMNS = (
    "subject",
    "body",
    "label",
    "subject_clean",
    "body_clean",
    "text",
)


class DataValidationError(ValueError):
    """Raised when an input record cannot safely be used for training."""


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style"}:
            self.ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self.ignored_depth:
            self.ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)

    def get_text(self) -> str:
        return " ".join(self.parts)


def canonicalize_text(value: str) -> str:
    """Normalize presentation while preserving addresses used to identify a message."""
    if not isinstance(value, str):
        return ""

    parser = _HTMLTextExtractor()
    parser.feed(value)
    parser.close()
    return re.sub(r"\s+", " ", parser.get_text().lower()).strip()


def clean_text(value: str) -> str:
    text = canonicalize_text(value)
    text = URL_PATTERN.sub(" URL ", text)
    text = EMAIL_PATTERN.sub(" EMAIL ", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_email(subject: str, body: str) -> Tuple[str, str, str]:
    subject_clean = clean_text(subject)
    body_clean = clean_text(body)
    text = f"subject: {subject_clean} body: {body_clean}"
    return subject_clean, body_clean, text


def canonicalize_email(subject: str, body: str) -> Tuple[str, str]:
    """Return duplicate-detection text without discarding URL or email identity."""
    return canonicalize_text(subject), canonicalize_text(body)


def _parse_label(value: object, row_number: int) -> int:
    label = str(value).strip()
    if label not in {"0", "1"}:
        raise DataValidationError(
            f"Row {row_number}: label must be 0 or 1; received {value!r}."
        )
    return int(label)


def validate_and_preprocess_record(
    record: Mapping[str, object], row_number: int
) -> Dict[str, object]:
    for field in ("subject", "body"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise DataValidationError(f"Row {row_number}: {field} is missing or blank.")

    label = _parse_label(record.get("label"), row_number)
    subject = str(record["subject"])
    body = str(record["body"])
    subject_clean, body_clean, text = preprocess_email(subject, body)
    if not subject_clean and not body_clean:
        raise DataValidationError(
            f"Row {row_number}: message is empty after preprocessing."
        )

    return {
        "subject": subject,
        "body": body,
        "label": label,
        "subject_clean": subject_clean,
        "body_clean": body_clean,
        "text": text,
    }


def iter_canonical_records(csv_file: TextIO) -> Iterator[Tuple[int, Dict[str, str]]]:
    """Yield canonical records; source-specific importers should target this schema."""
    reader = csv.DictReader(csv_file)
    if reader.fieldnames is None:
        raise DataValidationError("Input CSV is empty or has no header.")
    missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
    if missing:
        raise DataValidationError(
            f"Input CSV is missing required columns: {', '.join(missing)}."
        )
    for row_number, record in enumerate(reader, start=2):
        yield row_number, record


def preprocess_csv(input_path: Path, output_path: Path) -> Dict[str, int]:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"total": 0, "phishing": 0, "legitimate": 0}
    seen_messages: Dict[bytes, Tuple[int, int]] = {}
    temporary_path = None

    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as destination:
                temporary_path = Path(destination.name)
                writer = csv.DictWriter(
                    destination, fieldnames=OUTPUT_COLUMNS, lineterminator="\n"
                )
                writer.writeheader()

                for row_number, record in iter_canonical_records(source):
                    cleaned = validate_and_preprocess_record(record, row_number)
                    canonical_subject, canonical_body = canonicalize_email(
                        str(cleaned["subject"]), str(cleaned["body"])
                    )
                    message = f"{canonical_subject}\0{canonical_body}".encode("utf-8")
                    key = hashlib.sha256(message).digest()
                    if key in seen_messages:
                        first_row, first_label = seen_messages[key]
                        if cleaned["label"] != first_label:
                            raise DataValidationError(
                                f"Row {row_number}: message conflicts with row {first_row}; "
                                "the duplicate messages have different labels."
                            )
                        raise DataValidationError(
                            f"Row {row_number}: message duplicates row {first_row}."
                        )

                    seen_messages[key] = (row_number, cleaned["label"])
                    writer.writerow(cleaned)
                    counts["total"] += 1
                    if cleaned["label"] == 1:
                        counts["phishing"] += 1
                    else:
                        counts["legitimate"] += 1

        os.replace(temporary_path, output_path)
        return counts
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
