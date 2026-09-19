import pandas as pd
from pathlib import Path


# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "output" / "clean_sales_customer_product.csv"
OUTPUT_DIR = BASE_DIR / "output"


# Load cleaned dataset
df = pd.read_csv(INPUT_FILE)


# Create customer-level features
customer_features = df.groupby("Customer_ID").agg(
    Total_Transactions=("Transaction_ID", "count"),
    Total_Units=("Units", "sum"),
    Total_Net_Revenue=("Net_Revenue", "sum"),
    Average_Transaction_Value=("Net_Revenue", "mean"),
    Average_Discount=("Discount_Pct", "mean"),
    Return_Rate=("Return_Flag", "mean"),
    Unique_Products=("Product_ID", "nunique"),
    Unique_Categories=("Category_product", "nunique"),
    Unique_Channels=("Channel", "nunique"),
).reset_index()


# Add customer information
customer_info = df[
    [
        "Customer_ID",
        "Customer_City",
        "Age_Band",
        "Gender",
        "Acquisition_Channel",
        "Total_Orders",
        "Total_Revenue",
        "Churned",
    ]
].drop_duplicates("Customer_ID")


# Merge customer information with calculated features
customer_features = customer_features.merge(
    customer_info,
    on="Customer_ID",
    how="left"
)


# Convert return rate to percentage
customer_features["Return_Rate"] = (
    customer_features["Return_Rate"] * 100
)


# Round numerical values
numeric_columns = customer_features.select_dtypes(
    include="number"
).columns

customer_features[numeric_columns] = customer_features[
    numeric_columns
].round(2)


# Save customer-level feature dataset
output_file = OUTPUT_DIR / "customer_features.csv"

customer_features.to_csv(output_file, index=False)


print("===== FEATURE ENGINEERING =====")
print(f"Customers: {len(customer_features)}")
print(f"Features created: {len(customer_features.columns)}")
print(f"Output saved to: {output_file}")

print("\n===== FEATURES =====")
print(customer_features.columns.tolist())

print("\n===== FIRST 5 CUSTOMERS =====")
print(customer_features.head().to_string(index=False))