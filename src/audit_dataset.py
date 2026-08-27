import argparse
import csv
import hashlib
import json
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from itertools import zip_longest
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from preprocessing import canonicalize_email


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPOSITORY_ROOT / "data" / "processed" / "meajor_canonical.csv"
DEFAULT_METADATA = REPOSITORY_ROOT / "data" / "processed" / "meajor_metadata.jsonl"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "reports" / "meajor_data_quality.json"


def _fingerprint(subject: str, body: str) -> bytes:
    return hashlib.sha256(f"{subject}\0{body}".encode("utf-8")).digest()


def _length_summary(values: List[int]) -> Dict[str, float]:
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": round(statistics.fmean(ordered), 2),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _duplicate_summary(groups: Dict[bytes, List[int]]) -> Dict[str, int]:
    duplicate_groups = [counts for counts in groups.values() if sum(counts) > 1]
    conflicting_groups = [counts for counts in duplicate_groups if all(counts)]
    return {
        "unique_groups": len(groups),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_records_beyond_first": sum(
            sum(counts) - 1 for counts in duplicate_groups
        ),
        "conflicting_label_groups": len(conflicting_groups),
        "records_in_conflicting_groups": sum(map(sum, conflicting_groups)),
    }


def _add_group(groups: Dict[bytes, List[int]], key: bytes, label: int) -> None:
    groups[key][label] += 1


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        return str(path)


def audit_dataset(input_path: Path, metadata_path: Path) -> Dict[str, object]:
    input_path = Path(input_path)
    metadata_path = Path(metadata_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Canonical dataset not found: {input_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata JSONL not found: {metadata_path}")

    class_counts = Counter()
    source_counts = defaultdict(Counter)
    exact_groups = defaultdict(lambda: [0, 0])
    canonical_groups = defaultdict(lambda: [0, 0])
    training_eligible_groups = set()
    subject_char_lengths = []
    body_char_lengths = []
    subject_word_lengths = []
    body_word_lengths = []
    missing_subject = 0
    missing_body = 0
    invalid_labels = 0
    empty_after_canonicalization = 0
    metadata_alignment_errors = 0
    missing_source_metadata = 0
    very_long_subjects = 0
    very_long_bodies = 0
    very_short_nonblank_bodies = 0
    total = 0
    metadata_rows = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"subject", "body", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Canonical CSV must contain subject, body, and label columns.")

        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata_records: Iterable[Optional[str]] = metadata_file
            for canonical_index, pair in enumerate(
                zip_longest(reader, metadata_records), start=1
            ):
                row, metadata_line = pair
                if row is None:
                    metadata_rows += 1
                    metadata_alignment_errors += 1
                    continue

                total += 1
                subject = row.get("subject") or ""
                body = row.get("body") or ""
                subject_blank = not subject.strip()
                body_blank = not body.strip()
                missing_subject += subject_blank
                missing_body += body_blank

                label_text = (row.get("label") or "").strip()
                if label_text not in {"0", "1"}:
                    invalid_labels += 1
                    label = None
                else:
                    label = int(label_text)
                    class_counts[label] += 1

                subject_chars = len(subject.strip())
                body_chars = len(body.strip())
                subject_char_lengths.append(subject_chars)
                body_char_lengths.append(body_chars)
                subject_word_lengths.append(len(subject.split()))
                body_word_lengths.append(len(body.split()))
                very_long_subjects += subject_chars > 500
                very_long_bodies += body_chars > 100_000
                very_short_nonblank_bodies += 0 < body_chars < 10

                canonical_subject, canonical_body = canonicalize_email(subject, body)
                empty_after_canonicalization += not canonical_subject and not canonical_body
                if label is not None:
                    _add_group(exact_groups, _fingerprint(subject, body), label)
                    canonical_key = _fingerprint(canonical_subject, canonical_body)
                    _add_group(canonical_groups, canonical_key, label)
                    if not subject_blank and not body_blank and (
                        canonical_subject or canonical_body
                    ):
                        training_eligible_groups.add(canonical_key)

                if metadata_line is None:
                    metadata_alignment_errors += 1
                    continue
                metadata_rows += 1
                metadata = json.loads(metadata_line)
                if metadata.get("canonical_row") != canonical_index:
                    metadata_alignment_errors += 1
                source = metadata.get("source_fields", {}).get("source")
                if source is None or not str(source).strip():
                    missing_source_metadata += 1
                elif label is not None:
                    source_counts[str(source).strip()][label] += 1

    if not total:
        raise ValueError("Canonical dataset contains no records.")

    exact = _duplicate_summary(exact_groups)
    canonical = _duplicate_summary(canonical_groups)
    retained_after_deduplication = (
        canonical["unique_groups"] - canonical["conflicting_label_groups"]
    )
    conflicting_keys = {
        key for key, counts in canonical_groups.items() if all(counts)
    }
    retained_after_validation_and_deduplication = len(
        training_eligible_groups - conflicting_keys
    )
    labeled_total = class_counts[0] + class_counts[1]

    return {
        "input": _display_path(input_path),
        "metadata_input": _display_path(metadata_path),
        "records": {
            "total": total,
            "metadata_rows": metadata_rows,
            "class_counts": {
                "benign": class_counts[0],
                "phishing": class_counts[1],
            },
            "class_balance_percent": {
                "benign": round(class_counts[0] / labeled_total * 100, 4),
                "phishing": round(class_counts[1] / labeled_total * 100, 4),
            },
            "invalid_labels": invalid_labels,
        },
        "missing": {
            "blank_subject": missing_subject,
            "blank_body": missing_body,
            "empty_after_canonicalization": empty_after_canonicalization,
        },
        "duplicates": {
            "exact": exact,
            "canonicalized": canonical,
            "retained_after_recommended_deduplication": retained_after_deduplication,
            "retained_after_required_text_validation_and_deduplication": (
                retained_after_validation_and_deduplication
            ),
            "policy": (
                "Group by canonicalized subject and body; retain one record from each "
                "same-label group and exclude every conflicting-label group."
            ),
        },
        "lengths": {
            "subject_characters": _length_summary(subject_char_lengths),
            "body_characters": _length_summary(body_char_lengths),
            "subject_words": _length_summary(subject_word_lengths),
            "body_words": _length_summary(body_word_lengths),
        },
        "sources": {
            source: {
                "benign": counts[0],
                "phishing": counts[1],
                "total": counts[0] + counts[1],
                "percent_of_dataset": round(
                    (counts[0] + counts[1]) / labeled_total * 100, 4
                ),
                "phishing_percent": round(
                    counts[1] / (counts[0] + counts[1]) * 100, 4
                ),
            }
            for source, counts in sorted(source_counts.items())
        },
        "anomaly_counts": {
            "metadata_alignment_errors": metadata_alignment_errors,
            "missing_source_metadata": missing_source_metadata,
            "subjects_over_500_characters": very_long_subjects,
            "bodies_over_100000_characters": very_long_bodies,
            "nonblank_bodies_under_10_characters": very_short_nonblank_bodies,
        },
    }


def write_summary(summary: Dict[str, object], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            json.dump(summary, output_file, indent=2, ensure_ascii=False)
            output_file.write("\n")
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit canonical email data quality.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    summary = audit_dataset(args.input, args.metadata)
    write_summary(summary, args.output)
    records = summary["records"]
    duplicates = summary["duplicates"]
    print(f"Audit saved to {args.output}")
    print(f"Total: {records['total']}")
    print(f"Benign: {records['class_counts']['benign']}")
    print(f"Phishing: {records['class_counts']['phishing']}")
    print(
        "Retained after recommended deduplication: "
        f"{duplicates['retained_after_recommended_deduplication']}"
    )


if __name__ == "__main__":
    main()
