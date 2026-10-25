from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import ROOT, clean_data, load_data, split_data

MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

# Assumptions for the default retention simulation
RETENTION_COST = 300
SUCCESS_RATE = 0.30


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=42),
    }


def compare_models(models, X_train, y_train):
    """5-fold cross-validation on the training data only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["recall", "precision", "f1", "roc_auc"]
    rows = []
    for name, model in models.items():
        s = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        rows.append({
            "Model": name,
            "Recall": s["test_recall"].mean(),
            "Precision": s["test_precision"].mean(),
            "F1": s["test_f1"].mean(),
            "ROC-AUC": s["test_roc_auc"].mean(),
            "ROC-AUC std": s["test_roc_auc"].std(),
        })
    return pd.DataFrame(rows).set_index("Model").round(3)


def build_risk_table(cal_model, X_test, y_test, raw_df):
    """Per-customer churn risk plus the default cost-benefit result."""
    table = X_test.copy()
    table["Churn Probability"] = cal_model.predict_proba(X_test)[:, 1]
    table["Actual Churn"] = y_test.values
    table["Expected Benefit"] = (
        table["Churn Probability"] * SUCCESS_RATE * table["CLTV"] - RETENTION_COST
    )
    table["Target"] = table["Expected Benefit"] > 0
    table.insert(0, "CustomerID", raw_df.loc[X_test.index, "CustomerID"].values)
    return table


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    raw = load_data()
    clean = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(clean)

    models = build_models()

    print("Cross-validating models (this takes a minute or two)...")
    comparison = compare_models(models, X_train, y_train)
    print(comparison)
    comparison.to_csv(MODELS_DIR / "model_comparison.csv")

    # Final model: Logistic Regression (best recall, F1 and ROC-AUC)
    final = models["Logistic Regression"].fit(X_train, y_train)
    y_pred = final.predict(X_test)
    y_prob = final.predict_proba(X_test)[:, 1]
    print("\nHeld-out test set (Logistic Regression)")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))

    # Calibrated copy so predicted risk matches the real churn rate
    cal_model = CalibratedClassifierCV(final, method="sigmoid", cv=5)
    cal_model.fit(X_train, y_train)
    cal_probs = cal_model.predict_proba(X_test)[:, 1]
    print("\nActual churn rate:        ", round(y_test.mean(), 3))
    print("Average prediction (raw): ", round(y_prob.mean(), 3))
    print("Average prediction (calibrated):", round(cal_probs.mean(), 3))

    joblib.dump(final, MODELS_DIR / "churn_model.joblib")
    joblib.dump(cal_model, MODELS_DIR / "churn_model_calibrated.joblib")
    joblib.dump(X_train.columns.tolist(), MODELS_DIR / "feature_columns.joblib")

    risk_table = build_risk_table(cal_model, X_test, y_test, raw)
    risk_table.to_csv(DATA_DIR / "risk_table.csv", index=False)

    features = X_test.copy()
    features.insert(0, "CustomerID", raw.loc[X_test.index, "CustomerID"].values)
    features.to_csv(DATA_DIR / "test_features.csv", index=False)

    print("\nSaved models and tables.")
    print("Customers worth targeting:", int(risk_table["Target"].sum()),
          "of", len(risk_table))


if __name__ == "__main__":
    main()