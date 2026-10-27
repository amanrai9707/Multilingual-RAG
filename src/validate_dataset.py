"""CLI wrapper:  python -m src.validate_dataset --dataset X.xlsx --clean-dir data/clean"""
import argparse, sys
from src.dataset_io import load_dataset, validate, format_report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--clean-dir", default="data/clean")
    ap.add_argument("--min-per-category", type=int, default=25)
    ap.add_argument("--allow-missing-assamese", action="store_true", help="use before the Assamese questions are written")
    a = ap.parse_args(argv)
    res = validate(load_dataset(a.dataset), a.clean_dir, a.min_per_category, not a.allow_missing_assamese)
    print(format_report(res))
    return 1 if res["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
