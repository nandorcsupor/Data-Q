from pprint import pprint
from data_class import DataClass
import argparse
import pandas as pd

CSV_PATH = "data/data.csv"
OUTPUT_CSV_PATH = "data/report.csv"  # Define the path for the output report


def write_report_to_csv(report: dict, output_path: str):
    # Convert report to a DataFrame
    df_list = []
    
    for key, value in report.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                for index in sub_value:
                    df_list.append([key, sub_key, index])
        elif isinstance(value, list):  # This handles the "DUPLICATE_ROWS" section
            for item in value:
                df_list.append([key, 'Row Group', '-'.join(map(str, item))])
                
    df = pd.DataFrame(df_list, columns=["Category", "Column", "Row Indices"])

    # Write the DataFrame to a CSV file
    df.to_csv(output_path, index=False)



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
    
    # Write the report to a CSV file
    write_report_to_csv(report, OUTPUT_CSV_PATH)
    print(f"Report saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
