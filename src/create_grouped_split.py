import os
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# SUPPLYPRESCRIPT - GROUPED TRAIN/TEST SPLIT
# ============================================================

print("=" * 60)
print("SUPPLYPRESCRIPT GROUPED DATA SPLIT")
print("=" * 60)


# ------------------------------------------------------------
# 1. LOAD CLEANED DATASET
# ------------------------------------------------------------

INPUT_PATH = "data/validated/validated_dataset.csv"

df = pd.read_csv(INPUT_PATH)

print("\nDataset loaded successfully.")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ------------------------------------------------------------
# 2. TARGET
# ------------------------------------------------------------

TARGET = "Late_delivery_risk"
GROUP_COLUMN = "Order_Id"


# ------------------------------------------------------------
# 3. REMOVE LEAKAGE / UNNECESSARY COLUMNS
# ------------------------------------------------------------

columns_to_remove = [
    "Customer_Email",
    "Customer_Password",
    "Product_Description",
    "Product_Status",
    "Order_Item_Id",
    "Customer_Id",
    "Order_Customer_Id",
    "Product_Card_Id",
    "Order_Item_Cardprod_Id",
    "Category_Id",
    "Product_Category_Id",
    "Department_Id",
    "Product_Image",
    "Customer_Fname",
    "Customer_Lname",
    "Customer_Street",
    "Latitude",
    "Longitude",
    "Delivery_Status",
    "shipping_date_(DateOrders)",
    "Order_Status",
    "Days_for_shipping_(real)",
    "Days_for_shipment_(scheduled)",
]


# ------------------------------------------------------------
# 4. CREATE DATE FEATURES
# ------------------------------------------------------------

date_column = "order_date_(DateOrders)"

df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce"
)

df["order_year"] = df[date_column].dt.year
df["order_month"] = df[date_column].dt.month
df["order_day"] = df[date_column].dt.day
df["order_day_of_week"] = df[date_column].dt.dayofweek
df["order_week"] = df[date_column].dt.isocalendar().week.astype(int)


# Remove original date column
columns_to_remove.append(date_column)


# ------------------------------------------------------------
# 5. SEPARATE TARGET AND GROUP
# ------------------------------------------------------------

y = df[TARGET]
groups = df[GROUP_COLUMN]

X = df.drop(
    columns=[TARGET] + columns_to_remove
)


# ------------------------------------------------------------
# 6. GROUPED TRAIN/TEST SPLIT
# ------------------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[train_indices].copy()
X_test = X.iloc[test_indices].copy()

y_train = y.iloc[train_indices].copy()
y_test = y.iloc[test_indices].copy()


# ------------------------------------------------------------
# 7. ADD TARGET BACK
# ------------------------------------------------------------

train_df = X_train.copy()
train_df[TARGET] = y_train.values

test_df = X_test.copy()
test_df[TARGET] = y_test.values


# ------------------------------------------------------------
# 8. SAVE DATA
# ------------------------------------------------------------

OUTPUT_DIR = "data/model_grouped"

os.makedirs(OUTPUT_DIR, exist_ok=True)

train_path = os.path.join(
    OUTPUT_DIR,
    "train.csv"
)

test_path = os.path.join(
    OUTPUT_DIR,
    "test.csv"
)

train_df.to_csv(
    train_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)


# ------------------------------------------------------------
# 9. VALIDATION
# ------------------------------------------------------------

train_orders = set(
    groups.iloc[train_indices]
)

test_orders = set(
    groups.iloc[test_indices]
)

overlap = train_orders.intersection(test_orders)


print("\n" + "=" * 60)
print("GROUP SPLIT RESULTS")
print("=" * 60)

print(f"\nTraining rows: {len(train_df):,}")
print(f"Testing rows : {len(test_df):,}")

print(f"\nTraining orders: {len(train_orders):,}")
print(f"Testing orders : {len(test_orders):,}")

print(f"\nOrder overlap between train/test: {len(overlap)}")


# ------------------------------------------------------------
# 10. TARGET DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING TARGET DISTRIBUTION")
print("=" * 60)

print(
    y_train.value_counts()
    .sort_index()
)

print("\nPercentages:")

print(
    (y_train.value_counts(normalize=True)
     .sort_index() * 100)
     .round(2)
)


print("\n" + "=" * 60)
print("TEST TARGET DISTRIBUTION")
print("=" * 60)

print(
    y_test.value_counts()
    .sort_index()
)

print("\nPercentages:")

print(
    (y_test.value_counts(normalize=True)
     .sort_index() * 100)
     .round(2)
)


print("\n" + "=" * 60)
print("GROUPED SPLIT COMPLETED")
print("=" * 60)

print(f"\nTraining data: {train_path}")
print(f"Testing data : {test_path}")