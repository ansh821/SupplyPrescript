import pandas as pd
from pathlib import Path


class DataProfiler:

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None

    def load_data(self):
        """Load cleaned dataset."""

        try:
            self.df = pd.read_csv(self.file_path)

            print("Dataset loaded successfully.")
            print(
                f"Rows: {self.df.shape[0]:,}"
            )
            print(
                f"Columns: {self.df.shape[1]}"
            )

            return True

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def basic_information(self):

        print("\n" + "=" * 65)
        print("BASIC DATASET INFORMATION")
        print("=" * 65)

        print(
            f"\nRows: {self.df.shape[0]:,}"
        )

        print(
            f"Columns: {self.df.shape[1]}"
        )

        print(
            f"Memory usage: "
            f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )

    def column_profile(self):

        print("\n" + "=" * 65)
        print("COLUMN PROFILE")
        print("=" * 65)

        profile = []

        for column in self.df.columns:

            profile.append({
                "column": column,
                "dtype": str(self.df[column].dtype),
                "missing": int(
                    self.df[column].isnull().sum()
                ),
                "missing_%": round(
                    self.df[column].isnull().mean() * 100,
                    2
                ),
                "unique": int(
                    self.df[column].nunique(
                        dropna=True
                    )
                ),
                "unique_%": round(
                    self.df[column].nunique(
                        dropna=True
                    ) / len(self.df) * 100,
                    2
                )
            })

        profile_df = pd.DataFrame(profile)

        print(
            profile_df.to_string(index=False)
        )

        return profile_df

    def missing_value_analysis(self):

        print("\n" + "=" * 65)
        print("MISSING VALUE ANALYSIS")
        print("=" * 65)

        missing = self.df.isnull().sum()

        missing = missing[
            missing > 0
        ].sort_values(
            ascending=False
        )

        if missing.empty:

            print(
                "No missing values found."
            )

            return

        for column, count in missing.items():

            percentage = (
                count / len(self.df)
            ) * 100

            print(
                f"{column}: "
                f"{count:,} "
                f"({percentage:.2f}%)"
            )

    def constant_column_analysis(self):

        print("\n" + "=" * 65)
        print("CONSTANT COLUMN ANALYSIS")
        print("=" * 65)

        for column in self.df.columns:

            unique_values = self.df[
                column
            ].dropna().unique()

            if len(unique_values) <= 1:

                print(
                    f"\nColumn: {column}"
                )

                print(
                    f"Unique values: "
                    f"{unique_values}"
                )

    def categorical_analysis(self):

        print("\n" + "=" * 65)
        print("CATEGORICAL COLUMN ANALYSIS")
        print("=" * 65)

        categorical_columns = (
            self.df.select_dtypes(
                include=["object", "category"]
            ).columns
        )

        for column in categorical_columns:

            unique_count = (
                self.df[column]
                .nunique(dropna=True)
            )

            print(
                f"{column}: "
                f"{unique_count:,} unique values"
            )

    def numeric_summary(self):

        print("\n" + "=" * 65)
        print("NUMERIC COLUMN SUMMARY")
        print("=" * 65)

        numeric_df = self.df.select_dtypes(
            include="number"
        )

        if numeric_df.empty:

            print(
                "No numeric columns found."
            )

            return

        summary = numeric_df.describe().T

        summary["missing"] = (
            self.df[
                numeric_df.columns
            ].isnull().sum()
        )

        print(
            summary.to_string()
        )

    def run_profile(self):

        self.basic_information()

        self.column_profile()

        self.missing_value_analysis()

        self.constant_column_analysis()

        self.categorical_analysis()

        self.numeric_summary()


if __name__ == "__main__":

    file_path = (
        "data/validated/validated_dataset.csv"
    )

    profiler = DataProfiler(file_path)

    if profiler.load_data():

        profiler.run_profile()