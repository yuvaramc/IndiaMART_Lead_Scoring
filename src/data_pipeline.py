import pandas as pd
from pathlib import Path


# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"


# Create output folder if it does not exist
OUTPUT_DIR.mkdir(exist_ok=True)


# File paths
customer_file = DATA_DIR / "da_customer_data (1).csv"
sales_file = DATA_DIR / "da_sales_transactions (1).csv"
product_file = DATA_DIR / "da_product_catalogue (1).csv"


# Load datasets
customers = pd.read_csv(customer_file)
sales = pd.read_csv(sales_file)
products = pd.read_csv(product_file)


# Convert date columns
customers["First_Purchase_Date"] = pd.to_datetime(
    customers["First_Purchase_Date"]
)

customers["Last_Purchase_Date"] = pd.to_datetime(
    customers["Last_Purchase_Date"]
)

sales["Date"] = pd.to_datetime(sales["Date"])


# Merge sales with customer information
sales_customers = sales.merge(
    customers,
    on="Customer_ID",
    how="left"
)


# Merge product information
sales_full = sales_customers.merge(
    products,
    on="Product_ID",
    how="left",
    suffixes=("_sales", "_product")
)


# Clean duplicate/overlapping column names
sales_full = sales_full.rename(columns={
    "City_x": "Transaction_City",
    "City_y": "Customer_City"
})


# Calculate transaction revenue
sales_full["Transaction_Revenue"] = (
    sales_full["Units"] * sales_full["Unit_Price"]
)


# Calculate discount amount
sales_full["Discount_Amount"] = (
    sales_full["Transaction_Revenue"]
    * sales_full["Discount_Pct"]
    / 100
)


# Calculate revenue after discount
sales_full["Net_Revenue"] = (
    sales_full["Transaction_Revenue"]
    - sales_full["Discount_Amount"]
)


# Save cleaned dataset
output_file = OUTPUT_DIR / "clean_sales_customer_product.csv"

sales_full.to_csv(output_file, index=False)


print("Data pipeline completed successfully!")
print(f"Customers: {len(customers)}")
print(f"Sales transactions: {len(sales)}")
print(f"Products: {len(products)}")
print(f"Merged dataset rows: {len(sales_full)}")
print(f"Output saved to: {output_file}")