import pandas as pd
import numpy as np
from pathlib import Path


class DataQualityChecker:

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None
        self.errors = []
        self.warnings = []

    def load_data(self):
        """Load cleaned dataset."""

        try:
            self.df = pd.read_csv(self.file_path)

            print("Cleaned dataset loaded successfully.")
            print(f"Rows: {self.df.shape[0]}")
            print(f"Columns: {self.df.shape[1]}")

            return True

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def check_empty_dataset(self):
        """Check whether dataset is empty."""

        if self.df.empty:
            self.errors.append(
                "Dataset contains no records."
            )
        else:
            print("PASS: Dataset contains records.")

    def check_missing_values(self):
        """Check remaining missing values."""

        missing = self.df.isnull().sum()

        total_missing = missing.sum()

        if total_missing > 0:

            self.warnings.append(
                f"{total_missing} missing values remain."
            )

            print(
                f"WARNING: {total_missing} missing values remain."
            )

        else:
            print("PASS: No missing values.")

    def check_duplicates(self):
        """Check remaining duplicate records."""

        duplicates = self.df.duplicated().sum()

        if duplicates > 0:

            self.warnings.append(
                f"{duplicates} duplicate rows remain."
            )

            print(
                f"WARNING: {duplicates} duplicate rows remain."
            )

        else:
            print("PASS: No duplicate rows.")

    def check_infinite_values(self):
        """Check for infinite numeric values."""

        numeric_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        infinite_count = 0

        for column in numeric_columns:

            infinite_count += np.isinf(
                self.df[column]
            ).sum()

        if infinite_count > 0:

            self.errors.append(
                f"{infinite_count} infinite values found."
            )

            print(
                f"FAIL: {infinite_count} infinite values found."
            )

        else:
            print("PASS: No infinite values.")

    def check_column_names(self):
        """Check for empty or duplicate column names."""

        empty_names = [
            column
            for column in self.df.columns
            if not str(column).strip()
        ]

        duplicate_names = (
            self.df.columns.duplicated().sum()
        )

        if empty_names:

            self.errors.append(
                "Empty column names found."
            )

            print("FAIL: Empty column names found.")

        elif duplicate_names > 0:

            self.errors.append(
                "Duplicate column names found."
            )

            print("FAIL: Duplicate column names found.")

        else:

            print("PASS: Column names are valid.")

    def check_constant_columns(self):
        """Find columns containing only one unique value."""

        constant_columns = []

        for column in self.df.columns:

            if self.df[column].nunique(
                dropna=False
            ) <= 1:

                constant_columns.append(column)

        if constant_columns:

            self.warnings.append(
                f"Constant columns found: "
                f"{constant_columns}"
            )

            print(
                "WARNING: Constant columns:",
                constant_columns
            )

        else:

            print("PASS: No constant columns.")

    def check_numeric_columns(self):
        """Check numeric columns for invalid values."""

        numeric_columns = self.df.select_dtypes(
            include=np.number
        ).columns

        if len(numeric_columns) == 0:

            self.warnings.append(
                "No numeric columns found."
            )

            print("WARNING: No numeric columns found.")

        else:

            print(
                f"PASS: {len(numeric_columns)} "
                f"numeric columns detected."
            )

    def generate_summary(self):

        print("\n" + "=" * 55)
        print("DATA QUALITY REPORT")
        print("=" * 55)

        print(
            f"\nDataset size:"
            f" {self.df.shape[0]} rows × "
            f"{self.df.shape[1]} columns"
        )

        print(
            f"Errors: {len(self.errors)}"
        )

        print(
            f"Warnings: {len(self.warnings)}"
        )

        if self.errors:

            print("\nERRORS:")

            for error in self.errors:
                print(f"- {error}")

        if self.warnings:

            print("\nWARNINGS:")

            for warning in self.warnings:
                print(f"- {warning}")

        if not self.errors:

            print(
                "\nQUALITY STATUS: PASSED"
            )

        else:

            print(
                "\nQUALITY STATUS: FAILED"
            )

    def run_checks(self):

        print("\n" + "=" * 55)
        print("RUNNING DATA QUALITY CHECKS")
        print("=" * 55)

        self.check_empty_dataset()
        self.check_missing_values()
        self.check_duplicates()
        self.check_infinite_values()
        self.check_column_names()
        self.check_constant_columns()
        self.check_numeric_columns()

        self.generate_summary()


if __name__ == "__main__":

    file_path = (
        "data/validated/validated_dataset.csv"
    )

    checker = DataQualityChecker(file_path)

    if checker.load_data():

        checker.run_checks()