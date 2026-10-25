from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Telco_customer_churn.xlsx"

# Columns removed before modelling:
#  - Churn Score / Churn Reason / Churn Label leak the answer
#  - Count / Country / State are constant, CustomerID is just an ID
#  - Location columns have too many unique values for this model
DROP_COLS = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude",
    "Churn Label", "Churn Score", "Churn Reason",
]

YES_NO_COLS = [
    "Senior Citizen", "Partner", "Dependents", "Phone Service", "Paperless Billing",
    "Multiple Lines", "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies",
]

MULTI_COLS = ["Internet Service", "Contract", "Payment Method"]
TARGET = "Churn Value"


def load_data(path=DATA_PATH):
    """Read the raw Excel file."""
    return pd.read_excel(path)


def clean_data(df):
    """Return a fully numeric dataframe ready for modelling."""
    df = df.copy()

    # Blank Total Charges belong to brand-new customers (tenure 0), so they are 0
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0)

    df = df.drop(columns=DROP_COLS)

    # "No phone service" / "No internet service" are the same as "No"
    df = df.replace({"No phone service": "No", "No internet service": "No"})
    for col in YES_NO_COLS:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    df = pd.get_dummies(df, columns=MULTI_COLS, drop_first=True, dtype=int)
    return df


def split_data(df_clean, test_size=0.2, random_state=42):
    """Separate inputs from the target and make a stratified train/test split."""
    X = df_clean.drop(columns=[TARGET])
    y = df_clean[TARGET]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


if __name__ == "__main__":
    raw = load_data()
    clean = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(clean)
    print("Clean shape:", clean.shape)
    print("Train / test:", X_train.shape, X_test.shape)
    print("Churn rate:", round(y_train.mean(), 3), round(y_test.mean(), 3))
    print("Missing values:", int(clean.isnull().sum().sum()))