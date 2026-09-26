import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================
# INDIA MART B2B - CHECKPOINT 3
# Deep-Dive Analysis: Customer Segmentation
# ============================================

# File paths
INPUT_FILE = "output/customer_features.csv"
OUTPUT_DIR = "output/deep_dive"

# Create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load customer features
df = pd.read_csv(INPUT_FILE)

print("===== CHECKPOINT 3: DEEP-DIVE ANALYSIS =====")
print(f"Customers analyzed: {len(df)}")
print(f"Features available: {len(df.columns)}")

# ============================================
# 1. CREATE CUSTOMER SEGMENTS
# ============================================

# Revenue-based segmentation
revenue_q1 = df["Total_Net_Revenue"].quantile(0.33)
revenue_q2 = df["Total_Net_Revenue"].quantile(0.66)

def revenue_segment(value):
    if value <= revenue_q1:
        return "Low Revenue"
    elif value <= revenue_q2:
        return "Medium Revenue"
    else:
        return "High Revenue"

df["Revenue_Segment"] = df["Total_Net_Revenue"].apply(revenue_segment)

# Transaction-frequency segmentation
transaction_q1 = df["Total_Transactions"].quantile(0.33)
transaction_q2 = df["Total_Transactions"].quantile(0.66)

def frequency_segment(value):
    if value <= transaction_q1:
        return "Low Frequency"
    elif value <= transaction_q2:
        return "Medium Frequency"
    else:
        return "High Frequency"

df["Frequency_Segment"] = df["Total_Transactions"].apply(frequency_segment)

# ============================================
# 2. CUSTOMER SEGMENT SUMMARY
# ============================================

segment_summary = (
    df.groupby("Revenue_Segment")
    .agg(
        Customers=("Customer_ID", "count"),
        Total_Revenue=("Total_Net_Revenue", "sum"),
        Average_Revenue=("Total_Net_Revenue", "mean"),
        Average_Transactions=("Total_Transactions", "mean"),
        Average_Units=("Total_Units", "mean"),
        Average_Return_Rate=("Return_Rate", "mean"),
        Churned_Customers=("Churned", "sum")
    )
    .reset_index()
)

segment_summary["Churn_Rate"] = (
    segment_summary["Churned_Customers"]
    / segment_summary["Customers"]
    * 100
)

print("\n===== REVENUE SEGMENT SUMMARY =====")
print(segment_summary.to_string(index=False))

segment_summary.to_csv(
    f"{OUTPUT_DIR}/revenue_segment_summary.csv",
    index=False
)

# ============================================
# 3. ACQUISITION CHANNEL ANALYSIS
# ============================================

channel_summary = (
    df.groupby("Acquisition_Channel")
    .agg(
        Customers=("Customer_ID", "count"),
        Average_Revenue=("Total_Net_Revenue", "mean"),
        Average_Transactions=("Total_Transactions", "mean"),
        Average_Return_Rate=("Return_Rate", "mean"),
        Churned_Customers=("Churned", "sum")
    )
    .reset_index()
)

channel_summary["Churn_Rate"] = (
    channel_summary["Churned_Customers"]
    / channel_summary["Customers"]
    * 100
)

print("\n===== ACQUISITION CHANNEL ANALYSIS =====")
print(channel_summary.to_string(index=False))

channel_summary.to_csv(
    f"{OUTPUT_DIR}/acquisition_channel_analysis.csv",
    index=False
)

# ============================================
# 4. CHURN ANALYSIS
# ============================================

churn_summary = (
    df.groupby("Churned")
    .agg(
        Customers=("Customer_ID", "count"),
        Average_Revenue=("Total_Net_Revenue", "mean"),
        Average_Transactions=("Total_Transactions", "mean"),
        Average_Units=("Total_Units", "mean"),
        Average_Discount=("Average_Discount", "mean"),
        Average_Return_Rate=("Return_Rate", "mean"),
        Average_Orders=("Total_Orders", "mean")
    )
    .reset_index()
)

print("\n===== CHURN ANALYSIS =====")
print(churn_summary.to_string(index=False))

churn_summary.to_csv(
    f"{OUTPUT_DIR}/churn_analysis.csv",
    index=False
)

# ============================================
# 5. CITY ANALYSIS
# ============================================

city_summary = (
    df.groupby("Customer_City")
    .agg(
        Customers=("Customer_ID", "count"),
        Total_Revenue=("Total_Net_Revenue", "sum"),
        Average_Revenue=("Total_Net_Revenue", "mean"),
        Average_Transactions=("Total_Transactions", "mean"),
        Churned_Customers=("Churned", "sum")
    )
    .reset_index()
)

city_summary["Churn_Rate"] = (
    city_summary["Churned_Customers"]
    / city_summary["Customers"]
    * 100
)

print("\n===== CITY ANALYSIS =====")
print(city_summary.to_string(index=False))

city_summary.to_csv(
    f"{OUTPUT_DIR}/city_analysis.csv",
    index=False
)

# ============================================
# 6. VISUALIZATION - REVENUE SEGMENTS
# ============================================

plt.figure(figsize=(8, 5))

plt.bar(
    segment_summary["Revenue_Segment"],
    segment_summary["Total_Revenue"]
)

plt.title("Total Revenue by Customer Segment")
plt.xlabel("Revenue Segment")
plt.ylabel("Total Net Revenue")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/revenue_by_segment.png",
    dpi=150
)

plt.close()

# ============================================
# 7. VISUALIZATION - CHURN
# ============================================

plt.figure(figsize=(8, 5))

churn_labels = ["Active", "Churned"]
churn_values = [
    (df["Churned"] == 0).sum(),
    (df["Churned"] == 1).sum()
]

plt.bar(churn_labels, churn_values)

plt.title("Customer Churn Distribution")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/churn_distribution.png",
    dpi=150
)

plt.close()

# ============================================
# 8. SAVE COMPLETE DEEP-DIVE DATASET
# ============================================

df.to_csv(
    f"{OUTPUT_DIR}/customer_segments.csv",
    index=False
)

print("\n===== ANALYSIS COMPLETE =====")
print(f"Output folder: {os.path.abspath(OUTPUT_DIR)}")
print("Files created:")
print("- revenue_segment_summary.csv")
print("- acquisition_channel_analysis.csv")
print("- churn_analysis.csv")
print("- city_analysis.csv")
print("- revenue_by_segment.png")
print("- churn_distribution.png")
print("- customer_segments.csv")