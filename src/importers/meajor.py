"""Streaming adapter for the MeAJOR v2.0 CSV published on Zenodo.

The adapter requires only the documented subject, body, and label columns. Every
other source column is retained in a JSON Lines sidecar rather than being mixed
into the model's canonical input schema.
"""

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Dict

from preprocessing import DataValidationError, REQUIRED_COLUMNS


LABEL_MAPPING = {"0": 0, "1": 1}
CANONICAL_COLUMNS = ("subject", "body", "label")


def _map_label(value: object, row_number: int) -> int:
    source_label = str(value).strip()
    if source_label not in LABEL_MAPPING:
        raise DataValidationError(
            f"Row {row_number}: unsupported MeAJOR label {value!r}; expected 0 or 1."
        )
    return LABEL_MAPPING[source_label]


def import_meajor_csv(
    input_path: Path, canonical_output_path: Path, metadata_output_path: Path
) -> Dict[str, int]:
    """Map a MeAJOR CSV to canonical CSV plus row-aligned metadata JSONL."""
    input_path = Path(input_path)
    canonical_output_path = Path(canonical_output_path)
    metadata_output_path = Path(metadata_output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"MeAJOR CSV not found: {input_path}")
    if canonical_output_path == metadata_output_path:
        raise ValueError("Canonical and metadata outputs must be different files.")

    canonical_output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_temp = None
    metadata_temp = None
    counts = {"total": 0, "phishing": 0, "legitimate": 0}

    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if reader.fieldnames is None:
                raise DataValidationError("MeAJOR CSV is empty or has no header.")
            missing = [field for field in REQUIRED_COLUMNS if field not in reader.fieldnames]
            if missing:
                raise DataValidationError(
                    f"MeAJOR CSV is missing required columns: {', '.join(missing)}."
                )
            metadata_columns = [
                field for field in reader.fieldnames if field not in CANONICAL_COLUMNS
            ]

            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", newline="", dir=canonical_output_path.parent,
                prefix=f".{canonical_output_path.name}.", suffix=".tmp", delete=False,
            ) as canonical_file, tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=metadata_output_path.parent,
                prefix=f".{metadata_output_path.name}.", suffix=".tmp", delete=False,
            ) as metadata_file:
                canonical_temp = Path(canonical_file.name)
                metadata_temp = Path(metadata_file.name)
                writer = csv.DictWriter(
                    canonical_file, fieldnames=CANONICAL_COLUMNS, lineterminator="\n"
                )
                writer.writeheader()

                for row_number, row in enumerate(reader, start=2):
                    label = _map_label(row.get("label"), row_number)
                    writer.writerow(
                        {"subject": row.get("subject"), "body": row.get("body"), "label": label}
                    )
                    metadata = {
                        "canonical_row": counts["total"] + 1,
                        "source_row": row_number,
                        "source_fields": {field: row.get(field) for field in metadata_columns},
                    }
                    metadata_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                    counts["total"] += 1
                    counts["phishing" if label == 1 else "legitimate"] += 1

        os.replace(canonical_temp, canonical_output_path)
        os.replace(metadata_temp, metadata_output_path)
        return counts
    except Exception:
        for temporary_path in (canonical_temp, metadata_temp):
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        raise
