import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


INPUT_FILE = "data/validated/validated_dataset.csv"

TRAIN_FILE = "data/model/train.csv"
TEST_FILE = "data/model/test.csv"

TARGET = "Late_delivery_risk"


def load_data():
    df = pd.read_csv(INPUT_FILE)

    print("Dataset loaded successfully.")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


def create_date_features(df):
    """Convert order date into useful numerical features."""

    date_column = "order_date_(DateOrders)"

    if date_column in df.columns:

        df[date_column] = pd.to_datetime(
            df[date_column],
            errors="coerce"
        )

        df["order_year"] = (
            df[date_column].dt.year
        )

        df["order_month"] = (
            df[date_column].dt.month
        )

        df["order_day"] = (
            df[date_column].dt.day
        )

        df["order_day_of_week"] = (
            df[date_column].dt.dayofweek
        )

        df["order_week"] = (
            df[date_column].dt.isocalendar().week
            .astype("int64")
        )

        df = df.drop(
            columns=[date_column]
        )

    return df


def remove_unnecessary_columns(df):
    """
    Remove identifiers, private fields and
    fields that should not be used directly.
    """

    columns_to_remove = [

        # Private / masked fields
        "Customer_Email",
        "Customer_Password",

        # Empty / constant fields
        "Product_Description",
        "Product_Status",

        # Unique identifiers
        "Order_Item_Id",
        "Order_Id",
        "Customer_Id",
        "Order_Customer_Id",

        # Duplicate product/category identifiers
        "Product_Card_Id",
        "Order_Item_Cardprod_Id",
        "Category_Id",
        "Product_Category_Id",
        "Department_Id",

        # Image URL is not useful for this model
        "Product_Image",

        # Personal information
        "Customer_Fname",
        "Customer_Lname",
        "Customer_Street",

        # Geographic precision not required initially
        "Latitude",
        "Longitude",

        # Shipping outcome / post-outcome information
        "Delivery_Status",
        "shipping_date_(DateOrders)",

        # Order completion information
        "Order_Status",
        "Days_for_shipping_(real)",
        "Days_for_shipment_(scheduled)",
    ]

    existing_columns = [
        column
        for column in columns_to_remove
        if column in df.columns
    ]

    df = df.drop(
        columns=existing_columns
    )

    print("\nRemoved columns:")

    for column in existing_columns:
        print(f"- {column}")

    return df


def prepare_target(df):
    """Prepare binary target."""

    if TARGET not in df.columns:

        raise ValueError(
            f"Target column '{TARGET}' "
            "not found."
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    df = df.dropna(
        subset=[TARGET]
    )

    df[TARGET] = df[TARGET].astype(int)

    return df


def show_target_distribution(df):
    """Display target distribution."""

    print("\n" + "=" * 60)
    print("TARGET DISTRIBUTION")
    print("=" * 60)

    counts = df[TARGET].value_counts()

    percentages = (
        df[TARGET]
        .value_counts(normalize=True)
        * 100
    )

    for value in counts.index:

        print(
            f"Class {value}: "
            f"{counts[value]:,} "
            f"({percentages[value]:.2f}%)"
        )


def split_dataset(df):

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    train = X_train.copy()
    train[TARGET] = y_train

    test = X_test.copy()
    test[TARGET] = y_test

    return train, test


def save_datasets(train, test):

    Path(
        "data/model"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        TRAIN_FILE,
        index=False
    )

    test.to_csv(
        TEST_FILE,
        index=False
    )

    print("\nDatasets saved.")

    print(
        f"Training data: "
        f"{TRAIN_FILE}"
    )

    print(
        f"Testing data: "
        f"{TEST_FILE}"
    )


def main():

    print("\n" + "=" * 60)
    print("SUPPLYPRESCRIPT MODEL DATA PREPARATION")
    print("=" * 60)

    df = load_data()

    df = create_date_features(df)

    df = remove_unnecessary_columns(df)

    df = prepare_target(df)

    show_target_distribution(df)

    print("\nFinal feature count:")
    print(
        len(df.columns) - 1
    )

    train, test = split_dataset(df)

    print(
        f"\nTraining rows: "
        f"{len(train):,}"
    )

    print(
        f"Testing rows: "
        f"{len(test):,}"
    )

    save_datasets(
        train,
        test
    )

    print(
        "\nModel data preparation completed."
    )


if __name__ == "__main__":
    main()