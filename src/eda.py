import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class SupplyChainEDA:

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

    def dataset_overview(self):

        print("\n" + "=" * 60)
        print("DATASET OVERVIEW")
        print("=" * 60)

        print("\nColumns:")

        for column in self.df.columns:
            print(f"- {column}")

        print("\nData Types:")

        print(self.df.dtypes)

    def numerical_summary(self):

        print("\n" + "=" * 60)
        print("NUMERICAL SUMMARY")
        print("=" * 60)

        numeric_columns = self.df.select_dtypes(
            include="number"
        ).columns

        print(
            self.df[numeric_columns].describe().T
        )

    def categorical_summary(self):

        print("\n" + "=" * 60)
        print("CATEGORICAL SUMMARY")
        print("=" * 60)

        categorical_columns = (
            self.df.select_dtypes(
                include=["object", "category"]
            ).columns
        )

        for column in categorical_columns:

            print(
                f"\n--- {column} ---"
            )

            print(
                self.df[column]
                .value_counts()
                .head(10)
            )

    def plot_numeric_distributions(self):

        numeric_columns = self.df.select_dtypes(
            include="number"
        ).columns

        print("\nCreating numeric distributions...")

        for column in numeric_columns:

            plt.figure(figsize=(8, 5))

            self.df[column].dropna().plot(
                kind="hist",
                bins=30
            )

            plt.title(
                f"Distribution of {column}"
            )

            plt.xlabel(column)

            plt.ylabel("Frequency")

            plt.tight_layout()

            plt.show()

    def plot_categorical_distributions(self):

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

            # Avoid plotting extremely high-cardinality columns
            if unique_count > 20:
                continue

            plt.figure(figsize=(9, 5))

            self.df[column].value_counts(
                dropna=False
            ).head(20).plot(
                kind="bar"
            )

            plt.title(
                f"Distribution of {column}"
            )

            plt.xlabel(column)

            plt.ylabel("Count")

            plt.xticks(rotation=45)

            plt.tight_layout()

            plt.show()

    def correlation_analysis(self):

        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)

        numeric_df = self.df.select_dtypes(
            include="number"
        )

        correlation = numeric_df.corr()

        print(
            correlation.round(2)
        )

        plt.figure(figsize=(12, 9))

        plt.imshow(
            correlation,
            aspect="auto"
        )

        plt.colorbar()

        plt.xticks(
            range(len(correlation.columns)),
            correlation.columns,
            rotation=90
        )

        plt.yticks(
            range(len(correlation.columns)),
            correlation.columns
        )

        plt.title(
            "Numeric Feature Correlation"
        )

        plt.tight_layout()

        plt.show()

    def run_eda(self):

        self.dataset_overview()

        self.numerical_summary()

        self.categorical_summary()

        self.correlation_analysis()

        self.plot_numeric_distributions()

        self.plot_categorical_distributions()


if __name__ == "__main__":

    file_path = (
        "data/validated/validated_dataset.csv"
    )

    eda = SupplyChainEDA(file_path)

    if eda.load_data():

        eda.run_eda()