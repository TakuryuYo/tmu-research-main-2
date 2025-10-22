"""
Compute the mean and variance for each numeric column in the target CSV file.

By default the script analyses the camp exam CSV that lives in
``annotation_max-speed_results``. Use ``--csv`` to point to another file.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = (
    THIS_DIR.parent
    / "annotation_max-speed_results"
    / "annotation_multi_persons_11_sessions_1-2-3-4-5_Unknown_-2_id00_108samples_20251021_162914.csv"
)


def load_numeric_columns(csv_path: Path) -> dict[str, list[float]]:
    """Collect numeric values for each column; skip cells that cannot be parsed."""
    numeric_columns: dict[str, list[float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header row found in {csv_path}")

        for row in reader:
            for column, raw_value in row.items():
                if raw_value is None or raw_value == "":
                    continue
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
                numeric_columns.setdefault(column, []).append(value)
    return numeric_columns


def compute_statistics(numeric_columns: dict[str, list[float]]) -> list[tuple[str, float, float, int]]:
    """Return alphabetical stats: (column, mean, variance, count)."""
    results: list[tuple[str, float, float, int]] = []
    for column, values in numeric_columns.items():
        if not values:
            continue
        mean_value = statistics.fmean(values)
        variance_value = statistics.pvariance(values)
        results.append((column, mean_value, variance_value, len(values)))
    return sorted(results, key=lambda item: item[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute mean and variance for each numeric column in a CSV file."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to the CSV file (default: {DEFAULT_CSV.relative_to(THIS_DIR.parent)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to store the results as a CSV file.",
    )
    args = parser.parse_args()

    csv_path = args.csv if args.csv.is_absolute() else THIS_DIR.parent / args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    numeric_columns = load_numeric_columns(csv_path)
    if not numeric_columns:
        raise ValueError(f"No numeric columns found in {csv_path}")

    results = compute_statistics(numeric_columns)

    if args.output:
        output_path = args.output if args.output.is_absolute() else THIS_DIR / args.output
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["column", "mean", "variance", "count"])
            writer.writerows(results)
        print(f"Wrote statistics to {output_path}")
    else:
        print("column,mean,variance,count")
        for column, mean_value, variance_value, count in results:
            print(f"{column},{mean_value},{variance_value},{count}")


if __name__ == "__main__":
    main()
