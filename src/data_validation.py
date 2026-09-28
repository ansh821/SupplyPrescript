import pandas as pd
import numpy as np
from pathlib import Path


class DataValidator:
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        self.validation_report = {}

    def load_data(self):
        """Load CSV dataset."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Dataset loaded successfully.")
            print(f"Rows: {self.df.shape[0]}")
            print(f"Columns: {self.df.shape[1]}")
            return True

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def check_missing_values(self):
        """Check missing/null values."""
        missing = self.df.isnull().sum()

        missing = missing[missing > 0]

        self.validation_report["missing_values"] = missing.to_dict()

        print("\n--- Missing Values ---")

        if missing.empty:
            print("No missing values found.")
        else:
            print(missing)

    def check_duplicates(self):
        """Check duplicate rows."""
        duplicates = self.df.duplicated().sum()

        self.validation_report["duplicate_rows"] = int(duplicates)

        print("\n--- Duplicate Rows ---")
        print(f"Duplicate rows: {duplicates}")

    def check_data_types(self):
        """Display column data types."""
        print("\n--- Data Types ---")

        dtypes = self.df.dtypes

        self.validation_report["data_types"] = {
            column: str(dtype)
            for column, dtype in dtypes.items()
        }

        print(dtypes)

    def check_numeric_columns(self):
        """Check numeric columns for invalid values."""
        numeric_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        print("\n--- Numeric Column Validation ---")

        numeric_report = {}

        for column in numeric_columns:

            negative_values = (self.df[column] < 0).sum()

            infinite_values = np.isinf(
                self.df[column]
            ).sum()

            numeric_report[column] = {
                "negative_values": int(negative_values),
                "infinite_values": int(infinite_values)
            }

            print(
                f"{column}: "
                f"negative={negative_values}, "
                f"infinite={infinite_values}"
            )

        self.validation_report["numeric_validation"] = numeric_report

    def check_unique_values(self):
        """Check unique values for categorical columns."""
        print("\n--- Unique Value Summary ---")

        categorical_columns = self.df.select_dtypes(
            include=["object", "category"]
        ).columns

        unique_report = {}

        for column in categorical_columns:

            unique_count = self.df[column].nunique()

            unique_report[column] = int(unique_count)

            print(
                f"{column}: {unique_count} unique values"
            )

        self.validation_report["unique_values"] = unique_report

    def check_outliers(self):
        """Detect outliers using IQR method."""
        print("\n--- Outlier Detection ---")

        numeric_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        outlier_report = {}

        for column in numeric_columns:

            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)

            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = self.df[
                (self.df[column] < lower_bound) |
                (self.df[column] > upper_bound)
            ]

            count = len(outliers)

            outlier_report[column] = {
                "outlier_count": int(count),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound)
            }

            print(
                f"{column}: {count} outliers"
            )

        self.validation_report["outliers"] = outlier_report

    def generate_report(self):
        """Generate complete validation report."""

        print("\n" + "=" * 50)
        print("DATA VALIDATION REPORT")
        print("=" * 50)

        print(
            f"\nDataset Shape: "
            f"{self.df.shape[0]} rows × "
            f"{self.df.shape[1]} columns"
        )

        self.check_missing_values()
        self.check_duplicates()
        self.check_data_types()
        self.check_numeric_columns()
        self.check_unique_values()
        self.check_outliers()

        return self.validation_report

    def save_validated_data(self, output_path):
        """Save dataset after validation."""

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.df.to_csv(
            output_path,
            index=False
        )

        print(
            f"\nValidated dataset saved to:"
            f"\n{output_path}"
        )


if __name__ == "__main__":

    # Change this to your actual dataset path
    input_file = "data/raw/DataCoSupplyChainDataset1.csv"

    # Output location
    output_file = "data/validated/validated_dataset.csv"

    validator = DataValidator(input_file)

    if validator.load_data():

        validator.generate_report()

        validator.save_validated_data(
            output_file
        )