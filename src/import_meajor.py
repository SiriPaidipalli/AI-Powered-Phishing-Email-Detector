import argparse
from pathlib import Path
from typing import List, Optional

from importers.meajor import import_meajor_csv


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a manually downloaded MeAJOR CSV."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    counts = import_meajor_csv(args.input, args.output, args.metadata_output)
    print(f"Benign rows: {counts['legitimate']}")
    print(f"Phishing rows: {counts['phishing']}")
    print(f"Total rows: {counts['total']}")
    print(f"Metadata rows: {counts['total']}")


if __name__ == "__main__":
    main()
