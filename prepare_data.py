"""Command-line entry point for converting temporal CSV/TSV data to MAT."""

import argparse

from trsc import DATASETS, convert_csv_to_mat


def parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a temporal edge CSV/TSV file into TRSC MAT data"
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data")
    parser.add_argument("--time-slices", type=int)
    parser.add_argument("--source-column", default="From")
    parser.add_argument("--target-column", default="To")
    parser.add_argument("--value-column", default="Value")
    parser.add_argument("--timestamp-column", default="TimeStamp")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--has-header", type=parse_bool, default=True)
    parser.add_argument(
        "--binning",
        choices=("auto", "discrete", "equal-width", "equal-count"),
        default="auto",
    )
    parser.add_argument("--make-symmetric", type=parse_bool, default=False)
    parser.add_argument("--edge-life", type=int, default=0)
    parser.add_argument("--val-rate", type=float, default=0.1)
    parser.add_argument("--test-rate", type=float, default=0.2)
    args = parser.parse_args()

    value_column = None if args.value_column.lower() == "none" else args.value_column
    output = convert_csv_to_mat(
        args.input,
        args.dataset,
        args.output,
        time_slices=args.time_slices,
        source_column=args.source_column,
        target_column=args.target_column,
        value_column=value_column,
        timestamp_column=args.timestamp_column,
        delimiter=args.delimiter,
        has_header=args.has_header,
        binning=args.binning,
        make_symmetric=args.make_symmetric,
        edge_life=args.edge_life,
        val_rate=args.val_rate,
        test_rate=args.test_rate,
    )
    print(f"Saved: {output}")
    print(f"Node mapping: {output.with_name(output.stem + '_node_mapping.csv')}")


if __name__ == "__main__":
    main()

