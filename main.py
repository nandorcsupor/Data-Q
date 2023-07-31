from pprint import pprint
from data_class import DataClass
import argparse
import json

CSV_PATH = "data/data.csv"


def main():
    parser = argparse.ArgumentParser(description="Simple tool for checking csv files.")
    parser.add_argument(
        "--path", type=str, help="the path to the csv file", required=True
    )
    parser.add_argument(
        "--separator", type=str, help="the separator used in the csv file, default: ',' ", required=False, default=","
    )
    args = parser.parse_args()
    data_class = DataClass(args.path, args.separator)
    report = data_class.generate_report()

    pprint(report)


if __name__ == "__main__":
    main()
