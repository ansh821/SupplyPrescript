import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class DataPreprocessor:

    def __init__(self, file_path, target_column):
        self.file_path = Path(file_path)
        self.target_column = target_column

        self.df = None
        self.X = None
        self.y = None

        self.numeric_features = []
        self.categorical_features = []

        self.preprocessor = None

    def load_data(self):

        try:
            self.df = pd.read_csv(
                self.file_path
            )

            print("Dataset loaded successfully.")

            print(
                f"Rows: {self.df.shape[0]:,}"
            )

            print(
                f"Columns: {self.df.shape[1]}"
            )

            if self.target_column not in self.df.columns:

                print(
                    f"\nERROR: Target column "
                    f"'{self.target_column}' "
                    f"was not found."
                )

                print("\nAvailable columns:")

                for column in self.df.columns:
                    print(f"- {column}")

                return False

            return True

        except Exception as e:

            print(
                f"Error loading dataset: {e}"
            )

            return False

    def prepare_features(self):

        print(
            "\nPreparing features..."
        )

        # Separate target
        self.y = self.df[
            self.target_column
        ]

        self.X = self.df.drop(
            columns=[self.target_column]
        )

        # Remove obvious non-predictive/private columns
        columns_to_remove = []

        for column in self.X.columns:

            column_lower = column.lower()

            if (
                "password" in column_lower
                or "email" in column_lower
            ):

                columns_to_remove.append(
                    column
                )

        if columns_to_remove:

            self.X = self.X.drop(
                columns=columns_to_remove
            )

            print(
                "\nExcluded columns:"
            )

            for column in columns_to_remove:
                print(f"- {column}")

        # Identify numeric features
        self.numeric_features = (
            self.X.select_dtypes(
                include=np.number
            ).columns.tolist()
        )

        # Identify categorical features
        self.categorical_features = (
            self.X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
        )

        print(
            f"\nNumeric features: "
            f"{len(self.numeric_features)}"
        )

        print(
            f"Categorical features: "
            f"{len(self.categorical_features)}"
        )

    def build_preprocessor(self):

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    numeric_pipeline,
                    self.numeric_features
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    self.categorical_features
                )
            ]
        )

        print(
            "\nPreprocessing pipeline created."
        )

    def split_data(
        self,
        test_size=0.2,
        random_state=42
    ):

        X_train, X_test, y_train, y_test = (
            train_test_split(
                self.X,
                self.y,
                test_size=test_size,
                random_state=random_state
            )
        )

        print("\n" + "=" * 60)
        print("TRAIN / TEST SPLIT")
        print("=" * 60)

        print(
            f"Training rows: "
            f"{len(X_train):,}"
        )

        print(
            f"Testing rows: "
            f"{len(X_test):,}"
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )


if __name__ == "__main__":

    # IMPORTANT:
    # Replace this with the actual target
    # identified from your dataset.

    TARGET_COLUMN = "REPLACE_WITH_TARGET"

    FILE_PATH = (
        "data/validated/validated_dataset.csv"
    )

    processor = DataPreprocessor(
        FILE_PATH,
        TARGET_COLUMN
    )

    if processor.load_data():

        processor.prepare_features()

        processor.build_preprocessor()

        processor.split_data()