import argparse
from pathlib import Path

from preprocessing import preprocess_csv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPOSITORY_ROOT / "data" / "raw" / "combined.csv"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "data" / "processed" / "emails.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and preprocess canonical email data.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = preprocess_csv(args.input, args.output)
    print(f"Cleaned data saved to {args.output}")
    print(
        f"Samples: {counts['total']} | "
        f"phish={counts['phishing']} | legit={counts['legitimate']}"
    )


if __name__ == "__main__":
    main()
