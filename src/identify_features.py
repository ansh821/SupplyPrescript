import pandas as pd
from pathlib import Path


class FeatureIdentifier:

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None

    def load_data(self):
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

    def identify_columns(self):

        print("\n" + "=" * 65)
        print("FEATURE IDENTIFICATION")
        print("=" * 65)

        for column in self.df.columns:

            dtype = self.df[column].dtype
            unique = self.df[column].nunique()
            missing = self.df[column].isnull().sum()

            print(
                f"\nColumn: {column}"
            )

            print(
                f"  Type: {dtype}"
            )

            print(
                f"  Unique values: {unique:,}"
            )

            print(
                f"  Missing: {missing:,}"
            )

            if dtype == "object":

                values = (
                    self.df[column]
                    .dropna()
                    .astype(str)
                    .unique()[:5]
                )

                print(
                    f"  Examples: {list(values)}"
                )

    def identify_possible_targets(self):

        print("\n" + "=" * 65)
        print("POSSIBLE TARGET VARIABLES")
        print("=" * 65)

        keywords = [
            "target",
            "demand",
            "sales",
            "quantity",
            "order",
            "forecast",
            "stock",
            "inventory",
            "risk",
            "delay",
            "return",
            "profit",
            "cost",
            "price"
        ]

        matches = []

        for column in self.df.columns:

            column_lower = column.lower()

            for keyword in keywords:

                if keyword in column_lower:

                    matches.append(
                        (column, keyword)
                    )

                    break

        if matches:

            for column, keyword in matches:

                print(
                    f"- {column} "
                    f"(matched: {keyword})"
                )

        else:

            print(
                "No obvious target column found."
            )

    def identify_id_columns(self):

        print("\n" + "=" * 65)
        print("POSSIBLE ID COLUMNS")
        print("=" * 65)

        for column in self.df.columns:

            column_lower = column.lower()

            unique_ratio = (
                self.df[column]
                .nunique(dropna=True)
                / len(self.df)
            )

            if (
                "id" in column_lower
                or unique_ratio > 0.95
            ):

                print(
                    f"- {column} "
                    f"(unique ratio: "
                    f"{unique_ratio:.2%})"
                )

    def run(self):

        self.identify_columns()

        self.identify_possible_targets()

        self.identify_id_columns()


if __name__ == "__main__":

    file_path = (
        "data/validated/validated_dataset.csv"
    )

    identifier = FeatureIdentifier(
        file_path
    )

    if identifier.load_data():

        identifier.run()