import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

customers = pd.read_csv(
    "data/da_customer_data (1).csv"
)

sales = pd.read_csv(
    "data/da_sales_transactions (1).csv"
)

customers["First_Purchase_Date"] = pd.to_datetime(
    customers["First_Purchase_Date"]
)

customers["Last_Purchase_Date"] = pd.to_datetime(
    customers["Last_Purchase_Date"]
)

sales["Date"] = pd.to_datetime(
    sales["Date"]
)

# --------------------------------------------------
# 2. Create transaction features
# --------------------------------------------------

transaction_features = sales.groupby(
    "Customer_ID"
).agg(
    Transaction_Count=("Transaction_ID", "count"),
    Total_Units=("Units", "sum"),
    Average_Unit_Price=("Unit_Price", "mean"),
    Average_Discount=("Discount_Pct", "mean"),
    Return_Count=("Return_Flag", "sum")
).reset_index()

transaction_features["Return_Rate"] = (
    transaction_features["Return_Count"]
    / transaction_features["Transaction_Count"]
)

transaction_features["Average_Transaction_Value"] = (
    transaction_features["Total_Units"]
    * transaction_features["Average_Unit_Price"]
)

# --------------------------------------------------
# 3. Merge data
# --------------------------------------------------

df = customers.merge(
    transaction_features,
    on="Customer_ID",
    how="left"
)

numeric_transaction_features = [
    "Transaction_Count",
    "Total_Units",
    "Average_Unit_Price",
    "Average_Discount",
    "Return_Count",
    "Return_Rate",
    "Average_Transaction_Value"
]

df[numeric_transaction_features] = (
    df[numeric_transaction_features]
    .fillna(0)
)

# --------------------------------------------------
# 4. Recency
# --------------------------------------------------

analysis_date = sales["Date"].max()

df["Days_Since_Last_Purchase"] = (
    analysis_date - df["Last_Purchase_Date"]
).dt.days

# --------------------------------------------------
# 5. Features and target
# --------------------------------------------------

features = [
    "City",
    "Age_Band",
    "Gender",
    "Acquisition_Channel",
    "Total_Orders",
    "Total_Revenue",
    "Transaction_Count",
    "Total_Units",
    "Average_Unit_Price",
    "Average_Discount",
    "Return_Rate",
    "Average_Transaction_Value",
    "Days_Since_Last_Purchase"
]

target = "Churned"

X = df[features]
y = df[target]

categorical_features = [
    "City",
    "Age_Band",
    "Gender",
    "Acquisition_Channel"
]

numerical_features = [
    "Total_Orders",
    "Total_Revenue",
    "Transaction_Count",
    "Total_Units",
    "Average_Unit_Price",
    "Average_Discount",
    "Return_Rate",
    "Average_Transaction_Value",
    "Days_Since_Last_Purchase"
]

# --------------------------------------------------
# 6. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)

# --------------------------------------------------
# 7. Balanced Logistic Regression
# --------------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

# --------------------------------------------------
# 8. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# 9. Train
# --------------------------------------------------

model.fit(
    X_train,
    y_train
)

# --------------------------------------------------
# 10. Probability predictions
# --------------------------------------------------

y_probability = model.predict_proba(
    X_test
)[:, 1]

# --------------------------------------------------
# 11. AUC
# --------------------------------------------------

auc = roc_auc_score(
    y_test,
    y_probability
)

print("\n" + "=" * 60)
print("BALANCED LOGISTIC REGRESSION")
print("=" * 60)

print(
    "Test AUC:",
    round(auc, 4)
)

# --------------------------------------------------
# 12. Threshold analysis
# --------------------------------------------------

print("\nThreshold Analysis")
print("-" * 60)

thresholds = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60
]

for threshold in thresholds:

    predictions = (
        y_probability >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    if (fp + tn) > 0:
        false_positive_rate = (
            fp / (fp + tn)
        )
    else:
        false_positive_rate = 0

    if (tp + fn) > 0:
        recall = (
            tp / (tp + fn)
        )
    else:
        recall = 0

    print(
        f"Threshold {threshold:.2f} | "
        f"TP={tp} | FP={fp} | "
        f"TN={tn} | FN={fn} | "
        f"Recall={recall:.2f} | "
        f"FPR={false_positive_rate:.2f}"
    )

# --------------------------------------------------
# 13. Classification report at 0.50
# --------------------------------------------------

y_prediction = (
    y_probability >= 0.50
).astype(int)

print("\nClassification Report (0.50 threshold)")
print(
    classification_report(
        y_test,
        y_prediction,
        zero_division=0
    )
)

print("\nConfusion Matrix")
print(
    confusion_matrix(
        y_test,
        y_prediction
    )
)

# --------------------------------------------------
# 14. Feature importance
# --------------------------------------------------

feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = (
    model
    .named_steps["classifier"]
    .coef_[0]
)

importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Absolute_Importance": np.abs(coefficients)
})

importance = importance.sort_values(
    "Absolute_Importance",
    ascending=False
)

print("\nTop Feature Importance")
print("-" * 60)

print(
    importance.head(15)
    .to_string(index=False)
)

# --------------------------------------------------
# 15. Save outputs
# --------------------------------------------------

importance.to_csv(
    "output/feature_importance.csv",
    index=False
)

df.to_csv(
    "output/customer_model_dataset.csv",
    index=False
)

# --------------------------------------------------
# 16. Save test predictions
# --------------------------------------------------

test_results = X_test.copy()

test_results["Actual_Churn"] = y_test.values
test_results["Churn_Probability"] = y_probability

test_results.to_csv(
    "output/test_predictions.csv",
    index=False
)

print("\nOutput files created:")
print("output/feature_importance.csv")
print("output/customer_model_dataset.csv")
print("output/test_predictions.csv")

print("\nDeep-dive analysis completed.")