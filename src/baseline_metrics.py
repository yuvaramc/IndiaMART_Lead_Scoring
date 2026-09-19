import pandas as pd
from pathlib import Path


# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "output" / "clean_sales_customer_product.csv"


# Load cleaned dataset
df = pd.read_csv(INPUT_FILE)


# Basic metrics
total_transactions = len(df)
total_customers = df["Customer_ID"].nunique()
total_revenue = df["Net_Revenue"].sum()
average_transaction_value = df["Net_Revenue"].mean()
total_units = df["Units"].sum()
return_rate = df["Return_Flag"].mean() * 100
churn_rate = df["Churned"].mean() * 100


# Revenue by channel
revenue_by_channel = (
    df.groupby("Channel")["Net_Revenue"]
    .sum()
    .sort_values(ascending=False)
)


# Print results
print("===== BASELINE METRICS =====")
print(f"Total transactions: {total_transactions}")
print(f"Total customers: {total_customers}")
print(f"Total revenue: ₹{total_revenue:,.2f}")
print(f"Average transaction value: ₹{average_transaction_value:,.2f}")
print(f"Total units sold: {total_units:,}")
print(f"Return rate: {return_rate:.2f}%")
print(f"Churn rate: {churn_rate:.2f}%")


print("\n===== REVENUE BY CHANNEL =====")
print(revenue_by_channel)

# Save baseline metrics
metrics = pd.DataFrame({
    "Metric": [
        "Total transactions",
        "Total customers",
        "Total revenue",
        "Average transaction value",
        "Total units sold",
        "Return rate",
        "Churn rate"
    ],
    "Value": [
        total_transactions,
        total_customers,
        total_revenue,
        average_transaction_value,
        total_units,
        return_rate,
        churn_rate
    ]
})

metrics_file = BASE_DIR / "output" / "baseline_metrics.csv"
metrics.to_csv(metrics_file, index=False)

print(f"\nBaseline metrics saved to: {metrics_file}")