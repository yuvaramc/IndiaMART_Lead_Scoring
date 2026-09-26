import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

# ============================================================
# INDIA MART B2B - CHECKPOINT 3 DEEP-DIVE ANALYSIS
# Predictive modelling using customer churn as a proxy target
# ============================================================

print("=" * 70)
print("INDIAMART B2B - CHECKPOINT 3 DEEP-DIVE ANALYSIS")
print("=" * 70)

# ============================================================
# 1. BUSINESS OBJECTIVE
# ============================================================

print("""
Business Objective
------------------
The original project brief requests a lead scoring model for a
B2B marketplace sales team.

The supplied data pack contains customer, product and sales
transaction data, but does not contain explicit lead-level
variables such as enquiry type, response time, RFQ count,
industry or company size.

Therefore, this deep-dive uses customer churn as a proxy
predictive modelling exercise using the available data.
This limitation is documented rather than assuming unavailable
lead variables.
""")

# ============================================================
# 2. LOAD DATA
# ============================================================

customers = pd.read_csv(
    "data/da_customer_data (1).csv"
)

sales = pd.read_csv(
    "data/da_sales_transactions (1).csv"
)

print("\nCustomer records:", len(customers))
print("Sales transactions:", len(sales))

# ============================================================
# 3. TARGET DISTRIBUTION
# ============================================================

print("\nChurn Distribution")
print(customers["Churned"].value_counts())

churn_percentage = (
    customers["Churned"]
    .value_counts(normalize=True)
    * 100
)

print("\nChurn Percentage")
print(churn_percentage.round(2))

# ============================================================
# 4. DATE CONVERSION
# ============================================================

customers["First_Purchase_Date"] = pd.to_datetime(
    customers["First_Purchase_Date"]
)

customers["Last_Purchase_Date"] = pd.to_datetime(
    customers["Last_Purchase_Date"]
)

sales["Date"] = pd.to_datetime(
    sales["Date"]
)

# ============================================================
# 5. TRANSACTION FEATURES
# ============================================================

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

# ============================================================
# 6. MERGE CUSTOMER + TRANSACTION DATA
# ============================================================

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

# ============================================================
# 7. RECENCY FEATURE
# ============================================================

analysis_date = sales["Date"].max()

df["Days_Since_Last_Purchase"] = (
    analysis_date
    - df["Last_Purchase_Date"]
).dt.days

# ============================================================
# 8. MODEL FEATURES
# ============================================================

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

X = df[features]
y = df["Churned"]

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

# ============================================================
# 9. PREPROCESSING
# ============================================================

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

# ============================================================
# 10. LOGISTIC REGRESSION
# ============================================================

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

# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model.fit(
    X_train,
    y_train
)

# ============================================================
# 12. MODEL EVALUATION
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]

auc = roc_auc_score(
    y_test,
    probabilities
)

print("\nModel Performance")
print("-----------------")
print("Test AUC:", round(auc, 4))

predictions = (
    probabilities >= 0.50
).astype(int)

print("\nConfusion Matrix")
print(confusion_matrix(
    y_test,
    predictions
))

# ============================================================
# 13. THRESHOLD ANALYSIS
# ============================================================

print("\nThreshold Analysis")
print("------------------")

threshold_results = []

for threshold in [0.20, 0.30, 0.40, 0.50, 0.60]:

    prediction = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        prediction,
        labels=[0, 1]
    ).ravel()

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    threshold_results.append({
        "Threshold": threshold,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Recall": recall,
        "False_Positive_Rate": fpr
    })

threshold_df = pd.DataFrame(
    threshold_results
)

print(
    threshold_df.to_string(index=False)
)

threshold_df.to_csv(
    "output/deep_dive_threshold_analysis.csv",
    index=False
)

# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

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
    "Absolute_Importance": abs(coefficients)
})

importance = importance.sort_values(
    "Absolute_Importance",
    ascending=False
)

print("\nTop Features")
print("------------")

print(
    importance.head(15)
    .to_string(index=False)
)

importance.to_csv(
    "output/deep_dive_feature_importance.csv",
    index=False
)

# ============================================================
# 15. SAVE MODEL DATA
# ============================================================

df.to_csv(
    "output/deep_dive_customer_dataset.csv",
    index=False
)

# ============================================================
# 16. ASSUMPTIONS AND LIMITATIONS
# ============================================================

print("""
Assumptions and Limitations
---------------------------
1. Churned is used as the target because no lead conversion
   target is present in the supplied data.

2. Customer and transaction information is used to create
   behavioural features.

3. Logistic Regression is used as the requested predictive
   modelling technique.

4. Class weighting is used because only 3% of customers are
   labelled as churned.

5. The dataset contains only 200 customers and 6 churned
   observations, so model estimates should be interpreted
   cautiously.

6. The test set contains only one churned observation.
   Therefore threshold-level performance is not statistically
   robust.

7. This analysis should not be represented as the final
   5000-lead CRM scoring model because the supplied dataset
   does not contain 5000 new leads.
""")

print("\n" + "=" * 70)
print("CHECKPOINT 3 DEEP-DIVE COMPLETE")
print("=" * 70)