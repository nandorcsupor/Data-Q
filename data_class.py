from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

# - **UNIFORMITY** Is the data in the same format (per column)?
# - **DUPLICATES** Are no duplicates in the data?
# - **MISSING VALUES** Are there any null / missing values?
# - **OUTLIERS** Any outliers in the data (per column)?

column_name = str

class DataClass:
    def __init__(self, path: str, separator: str = ";") -> None:
        self.df: pd.DataFrame = pd.read_csv(path, sep=separator, encoding="ISO-8859-1")
        self.df.columns = self.df.columns.str.strip()
        df = pd.read_csv(path, sep=separator)
        df = df[df.columns[0]].str.split(";", expand=True)
        df.columns = ['Make','Model','Vehicle Class','Engine Size(L)','Cylinders','Transmission',
                    'Fuel Type','Fuel Consumption City (L/100 km)',
                    'Fuel Consumption Hwy (L/100 km)','Fuel Consumption Comb (L/100 km)',
                    'Fuel Consumption Comb (mpg)','CO2 Emissions(g/km)']
        self.df = df

        # Remove leading/trailing whitespaces from column names
        self.df.columns = self.df.columns.str.strip()
        
        # Map month abbreviations to month numbers
        month_map = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 
            'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
        }
        
        # Columns that contain month abbreviations
        month_columns = [
            'Engine Size(L)', 'Fuel Consumption City (L/100 km)', 
            'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)'
        ]
        
        # Replace month abbreviations with month numbers
        for col in month_columns:
            self.df[col] = self.df[col].replace(month_map, regex=True)
            self.df[col] = self.df[col].str.replace(r'[^\d\.]', '', regex=True)
            self.df[col] = self.df[col].replace('', np.nan)
            self.df[col] = self.df[col].astype(float)

        # For MISSING VALUES
        # Make a copy of the original dataframe
        self.df_copy = self.df.copy()

        # Replace any empty or whitespace strings with NaN
        self.df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        self.missing_values = self.check_missing_values()

        # Restore the original dataframe
        self.df = self.df_copy

        self.df = self.df.fillna(value={col: 0 for col in month_columns})

    @staticmethod
    def is_number(x) -> bool:
        try:
            float(x)
            return True
        except ValueError:
            return False

    def check_uniformity(self) -> Dict[str, List[int]]:
        # Define expected types
        expected_dtypes = {
            'Make': str,
            'Model': str,
            'Vehicle Class': str,
            'Engine Size(L)': float,
            'Cylinders': int,
            'Transmission': str,
            'Fuel Type': str,
            'Fuel Consumption City (L/100 km)': float,
            'Fuel Consumption Hwy (L/100 km)': float,
            'Fuel Consumption Comb (L/100 km)': float,
            'Fuel Consumption Comb (mpg)': int,
            'CO2 Emissions(g/km)': int
        }

        # Initialize dictionary to store non-uniform data
        non_uniform_data = {}

        # Check each column
        for col, dtype in expected_dtypes.items():
            # Check if the data type is numeric
            if np.issubdtype(dtype, np.number):
                mask = self.df[col].apply(lambda x: isinstance(x, str) and not x.isnumeric())
            else:
                mask = self.df[col].apply(lambda x: self.is_number(x))
            # Get indices of non-uniform data
            non_uniform_indices = self.df[mask].index.tolist()
            if non_uniform_indices:
                non_uniform_data[col] = non_uniform_indices

        return non_uniform_data

    def check_duplicates(self) -> List[Tuple[int]]:
        # Return a list of tuples of row indexes where each tuple represents a duplicate group
        duplicates = self.df[self.df.duplicated(keep=False)]
        grouped = duplicates.groupby(list(duplicates.columns))
        idx = [indices for _, indices in grouped.groups.items()]
        return idx

    def check_missing_values(self) -> Dict[str, List[int]]:
        # Return a dictionary where keys are column names and values are lists of row indexes containing missing values in that column
        missing = {col: self.df[self.df[col].isnull()].index.tolist() for col in self.df.columns}
        return missing

    def check_outliers(self) -> Dict[str, List[int]]:
        # Outliers are defined by the 1.5 IQR method.
        outliers = {}
        for col in self.df.select_dtypes(include=np.number).columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_in_col = self.df.loc[(self.df[col] < lower_bound) | (self.df[col] > upper_bound), col].index.tolist()
            if outliers_in_col:
                outliers[col] = outliers_in_col
        return outliers

    def generate_report(self) -> Dict[str, Any]:
        report = {
            "DUPLICATE_ROWS": self.check_duplicates(),
            "UNIFORMITY": self.check_uniformity(),
            "MISSING_VALUE_ROWS": self.missing_values,
            "OUTLIERS": self.check_outliers(),
        }
        return report
