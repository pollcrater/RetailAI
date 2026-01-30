from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Quickly inspect a CSV or Parquet dataset")
    parser.add_argument("path", type=Path, help="Path to a .csv or .parquet file")
    parser.add_argument("--rows", type=int, default=5, help="Number of rows to print")
    args = parser.parse_args()

    path: Path = args.path
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise SystemExit("Supported formats: .csv, .parquet")

    print(f"shape: {df.shape}")
    print("\ncolumns:")
    print(list(df.columns))
    print("\nhead:")
    print(df.head(args.rows).to_string(index=False))

    print("\ndtypes:")
    print(df.dtypes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
