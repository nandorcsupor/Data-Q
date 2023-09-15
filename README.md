# DQC [BACKEND] Tech Challenge

## About the data file

The dataset `../data/data.csv` contains official records of CO2 emissions data by various cars of different features.
There are total 7385 rows and 12 columns.

Take a look at `../data/Data Description.csv` for further info regarding the columns.

Due to manual input errors, some values are either missing, have outliers, ...

## Your task

Please write a python tool to analyze the data quality across different dimensions:

- **UNIFORMITY** Is the data in the same format (per column)?
- **DUPLICATES** Are no duplicates in the data?
- **MISSING VALUES** Are there any null / missing values?
- **OUTLIERS** Any outliers in the data (per column)?

* We have created a simple framework with function stubs.

  - It consists of 2 parts. The `main.py` file which can be used as a command line
    tool by running: `python main.py --path /path/to/your/data.csv`
  - The DataClass class which holds the data read in from the csv and creates a layer of
    abstraction to simplify the interaction with raw csv files.

* Your task is to implement the functionality of the individual methods in `data_class` namely:
  - check_uniformity
  - check_duplicates
  - check_missing_values
  - check_outliers

# Run the code
python main.py --path ./data/data.csv
