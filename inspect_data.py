import pandas as pd
import os

data_folder = "data"

for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        path = os.path.join(data_folder, file)
        df = pd.read_csv(path)

        print("\n" + "=" * 60)
        print(file)
        print("=" * 60)

        print("Rows:", len(df))
        print("Columns:", len(df.columns))

        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))

        print("\nMissing values:")
        print(df.isnull().sum())