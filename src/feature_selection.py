import pandas as pd
from pathlib import Path


class FeatureSelector:

    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.df = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)

            print("Dataset loaded successfully.")
            print(f"Rows: {self.df.shape[0]:,}")
            print(f"Columns: {self.df.shape[1]}")

            return True

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def identify_id_columns(self):
        """Identify columns that mainly represent IDs."""

        id_columns = []

        for column in self.df.columns:

            column_lower = column.lower()

            unique_ratio = (
                self.df[column]
                .nunique(dropna=True)
                / len(self.df)
            )

            if (
                column_lower.endswith("_id")
                or column_lower == "id"
                or "password" in column_lower
                or "email" in column_lower
            ):
                id_columns.append(column)

            elif unique_ratio > 0.98:
                id_columns.append(column)

        return list(dict.fromkeys(id_columns))

    def identify_numeric_features(self, excluded):
        """Identify numerical features."""

        return [
            column
            for column in self.df.select_dtypes(
                include="number"
            ).columns
            if column not in excluded
        ]

    def identify_categorical_features(self, excluded):
        """Identify categorical features."""

        return [
            column
            for column in self.df.select_dtypes(
                include=["object", "category"]
            ).columns
            if column not in excluded
        ]

    def identify_target_candidates(self):
        """Find possible target variables."""

        keywords = [
            "demand",
            "sales",
            "quantity",
            "stock",
            "inventory",
            "order",
            "forecast",
            "risk",
            "delay",
            "return",
            "profit",
            "cost"
        ]

        candidates = []

        for column in self.df.columns:

            column_lower = column.lower()

            for keyword in keywords:

                if keyword in column_lower:

                    candidates.append(column)
                    break

        return list(dict.fromkeys(candidates))

    def generate_report(self):

        print("\n" + "=" * 65)
        print("FEATURE SELECTION REPORT")
        print("=" * 65)

        id_columns = self.identify_id_columns()

        numeric_features = self.identify_numeric_features(
            id_columns
        )

        categorical_features = (
            self.identify_categorical_features(
                id_columns
            )
        )

        target_candidates = (
            self.identify_target_candidates()
        )

        print("\n--- ID / EXCLUDED COLUMNS ---")

        for column in id_columns:
            print(f"- {column}")

        print("\n--- NUMERICAL FEATURES ---")

        for column in numeric_features:
            print(f"- {column}")

        print("\n--- CATEGORICAL FEATURES ---")

        for column in categorical_features:
            print(f"- {column}")

        print("\n--- POSSIBLE TARGET COLUMNS ---")

        for column in target_candidates:
            print(f"- {column}")

        print("\n" + "=" * 65)

        print(
            f"ID columns: {len(id_columns)}"
        )

        print(
            f"Numerical features: "
            f"{len(numeric_features)}"
        )

        print(
            f"Categorical features: "
            f"{len(categorical_features)}"
        )

        print(
            f"Target candidates: "
            f"{len(target_candidates)}"
        )


if __name__ == "__main__":

    file_path = (
        "data/validated/validated_dataset.csv"
    )

    selector = FeatureSelector(file_path)

    if selector.load_data():

        selector.generate_report()