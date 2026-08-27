import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from preprocessing import OUTPUT_COLUMNS, canonicalize_email, validate_and_preprocess_record


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPOSITORY_ROOT / "data" / "processed" / "meajor_canonical.csv"
DEFAULT_METADATA = REPOSITORY_ROOT / "data" / "processed" / "meajor_metadata.jsonl"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "processed"
DEFAULT_SUMMARY = REPOSITORY_ROOT / "reports" / "meajor_split_summary.json"
SOURCE_SPLITS = {"trec5": "train", "trec6": "dev", "trec7": "test"}


def _fingerprint(subject: str, body: str) -> bytes:
    canonical_subject, canonical_body = canonicalize_email(subject, body)
    message = f"{canonical_subject}\0{canonical_body}".encode("utf-8")
    return hashlib.sha256(message).digest()


def _read_pair(row, metadata_line: Optional[str], canonical_row: int):
    if row is None or metadata_line is None:
        raise ValueError("Canonical CSV and metadata JSONL have different row counts.")
    metadata = json.loads(metadata_line)
    if metadata.get("canonical_row") != canonical_row:
        raise ValueError(
            f"Metadata alignment error at canonical row {canonical_row}."
        )
    return metadata


def _scan_groups(input_path: Path, metadata_path: Path):
    groups = {}
    counts = Counter()
    with input_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"subject", "body", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Canonical CSV must contain subject, body, and label columns.")

        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            for canonical_row, (row, metadata_line) in enumerate(
                zip_longest(reader, metadata_file), start=1
            ):
                metadata = _read_pair(row, metadata_line, canonical_row)
                subject = row.get("subject") or ""
                body = row.get("body") or ""
                label_text = (row.get("label") or "").strip()
                if label_text not in {"0", "1"}:
                    raise ValueError(
                        f"Canonical row {canonical_row}: label must be 0 or 1."
                    )
                label = int(label_text)
                key = _fingerprint(subject, body)
                group = groups.setdefault(
                    key, {"labels": [0, 0], "representative": None, "sources": set()}
                )
                group["labels"][label] += 1
                source = metadata.get("source_fields", {}).get("source")
                source = str(source).strip() if source is not None else ""
                if source not in SOURCE_SPLITS:
                    raise ValueError(
                        f"Canonical row {canonical_row}: unsupported source {source!r}."
                    )
                group["sources"].add(source)
                counts["input_rows"] += 1

                if not subject.strip():
                    counts["blank_subject_rows"] += 1
                if not body.strip():
                    counts["blank_body_rows"] += 1
                if subject.strip() and body.strip() and group["representative"] is None:
                    group["representative"] = canonical_row

    duplicate_groups = 0
    duplicate_records = 0
    conflicting_groups = 0
    conflicting_records = 0
    cross_source_groups = 0
    cross_source_records = 0
    for group in groups.values():
        size = sum(group["labels"])
        if size > 1:
            duplicate_groups += 1
            duplicate_records += size - 1
        if all(group["labels"]):
            conflicting_groups += 1
            conflicting_records += size
        if size > 1 and len(group["sources"]) > 1:
            cross_source_groups += 1
            cross_source_records += size

    counts.update(
        {
            "unique_groups": len(groups),
            "duplicate_groups": duplicate_groups,
            "duplicate_records_beyond_first": duplicate_records,
            "conflicting_label_groups": conflicting_groups,
            "records_in_conflicting_groups": conflicting_records,
            "cross_source_duplicate_groups": cross_source_groups,
            "records_in_cross_source_duplicate_groups": cross_source_records,
        }
    )
    return groups, counts


def _open_temporary_outputs(output_dir: Path):
    handles = {}
    paths = {}
    for split in SOURCE_SPLITS.values():
        for kind, suffix in (("data", ".csv"), ("metadata", "_metadata.jsonl")):
            final_path = output_dir / f"{split}{suffix}"
            handle = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                newline="" if kind == "data" else None,
                dir=output_dir,
                prefix=f".{final_path.name}.",
                suffix=".tmp",
                delete=False,
            )
            handles[(split, kind)] = handle
            paths[(split, kind)] = (Path(handle.name), final_path)
    return handles, paths


def prepare_splits(
    input_path: Path,
    metadata_path: Path,
    output_dir: Path,
) -> Dict[str, object]:
    input_path = Path(input_path)
    metadata_path = Path(metadata_path)
    output_dir = Path(output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Canonical dataset not found: {input_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata JSONL not found: {metadata_path}")

    groups, preparation_counts = _scan_groups(input_path, metadata_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    handles, output_paths = _open_temporary_outputs(output_dir)
    split_counts = {
        split: {"total": 0, "benign": 0, "phishing": 0, "sources": Counter()}
        for split in SOURCE_SPLITS.values()
    }

    try:
        writers = {}
        for split in SOURCE_SPLITS.values():
            writer = csv.DictWriter(
                handles[(split, "data")],
                fieldnames=OUTPUT_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()
            writers[split] = writer

        with input_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            with metadata_path.open("r", encoding="utf-8") as metadata_file:
                for canonical_row, (row, metadata_line) in enumerate(
                    zip_longest(reader, metadata_file), start=1
                ):
                    metadata = _read_pair(row, metadata_line, canonical_row)
                    subject = row.get("subject") or ""
                    body = row.get("body") or ""
                    group = groups[_fingerprint(subject, body)]
                    if all(group["labels"]) or group["representative"] != canonical_row:
                        continue

                    cleaned = validate_and_preprocess_record(row, canonical_row + 1)
                    source = metadata.get("source_fields", {}).get("source")
                    source = str(source).strip() if source is not None else ""
                    if source not in SOURCE_SPLITS:
                        raise ValueError(
                            f"Canonical row {canonical_row}: unsupported source {source!r}."
                        )
                    split = SOURCE_SPLITS[source]
                    writers[split].writerow(cleaned)
                    split_counts[split]["total"] += 1
                    split_counts[split]["phishing" if cleaned["label"] else "benign"] += 1
                    split_counts[split]["sources"][source] += 1
                    retained_metadata = dict(metadata)
                    retained_metadata["split"] = split
                    retained_metadata["split_row"] = split_counts[split]["total"]
                    handles[(split, "metadata")].write(
                        json.dumps(retained_metadata, ensure_ascii=False) + "\n"
                    )

        for handle in handles.values():
            handle.close()
        for temporary_path, final_path in output_paths.values():
            os.replace(temporary_path, final_path)

        retained_total = sum(counts["total"] for counts in split_counts.values())
        preparation_counts["groups_excluded_for_blank_subject_or_body"] = (
            preparation_counts["unique_groups"]
            - preparation_counts["conflicting_label_groups"]
            - retained_total
        )
        return {
            "input": "data/processed/meajor_canonical.csv",
            "metadata_input": "data/processed/meajor_metadata.jsonl",
            "deduplication": dict(preparation_counts),
            "splits": {
                split: {
                    "total": counts["total"],
                    "benign": counts["benign"],
                    "phishing": counts["phishing"],
                    "sources": dict(sorted(counts["sources"].items())),
                    "data_file": f"data/processed/{split}.csv",
                    "metadata_file": f"data/processed/{split}_metadata.jsonl",
                }
                for split, counts in split_counts.items()
            },
            "retained_total": retained_total,
        }
    except Exception:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        for temporary_path, _ in output_paths.values():
            temporary_path.unlink(missing_ok=True)
        raise


def write_summary(summary: Dict[str, object], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            json.dump(summary, output_file, indent=2, ensure_ascii=False)
            output_file.write("\n")
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate MeAJOR and build source-disjoint model datasets."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    summary = prepare_splits(args.input, args.metadata, args.output_dir)
    write_summary(summary, args.summary)
    print(f"Split summary saved to {args.summary}")
    for split, counts in summary["splits"].items():
        print(
            f"{split}: total={counts['total']} benign={counts['benign']} "
            f"phishing={counts['phishing']} sources={counts['sources']}"
        )
    print(f"Retained total: {summary['retained_total']}")


if __name__ == "__main__":
    main()
