import pandas as pd
import os

def run_etl_pipeline():
    print("Starting ETL pipeline...")

    # load raw data
    raw_path = "data/raw/raw_sales_data.csv"
    df = pd.read_csv(raw_path)

    # inspect the initial data
    print("\n--- INITIAL DATA STATE ---")
    print(f"Total rows: {len(df)}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Missing values per column:")
    print(df.isna().sum())

    # clean the data
    print("\n--- APPLYING TRANSFORMATIONS ---")
    df = df.drop_duplicates() # remove duplicate rows
    print(f"Removed duplicates. New total rows: {len(df)}")
    df['date'] = pd.to_datetime(df['date'], format='mixed') # convert date column to datetime
    df['sales_volume'] = df['sales_volume'].ffill() # forward fill missing sales_volume values

    # inspect the final state
    print("\n--- FINAL DATA STATE ---")
    print(f"Total Rows: {len(df)}")
    print(f"Data Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isna().sum()}")

    # save the cleaned data
    print("\n--- SAVING CLEANED DATA ---")
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    processed_path = os.path.join(output_dir, "cleaned_sales_data.csv")
    df.to_csv(processed_path, index=False)
    print(f"SUCCESS: Cleaned dataset saved to {processed_path}")

if __name__ == "__main__":
    run_etl_pipeline()