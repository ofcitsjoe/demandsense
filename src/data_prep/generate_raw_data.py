import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_raw_dataset():
    np.random.seed(42) 

    # generate dates over 60 days
    start_date = datetime(2026, 5, 1)
    date_list = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(60)]

    # store and product configuration
    stores = ['STORE_001', 'STORE_002']
    products = ['PROD_A', 'PROD_B']

    data = []
    for date in date_list:
        for store in stores:
            for product in products:
                sales = np.random.randint(10, 150)
                data.append([date, store, product, sales])

    df = pd.DataFrame(data, columns=['date', 'store_id', 'product_id', 'sales_volume'])

    # 1. Inject missing sales numbers (Nulls)
    df.loc[df.sample(frac=0.05).index, 'sales_volume'] = np.nan
    
    # 2. Inject duplicate rows (Double scans)
    duplicates = df.sample(n=10, random_state=42).copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    # 3. Mess up some date types by turning them into raw text phrases occasionally
    df.loc[0, 'date'] = 'May-01-2026'
    
    return df

if __name__ == "__main__":
    raw_df = create_raw_dataset()

    # Save the raw dataset to a CSV file
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "raw_sales_data.csv")
    raw_df.to_csv(output_path, index=False)

    print(f"SUCCESS:Raw dataset generated and saved to {output_path}")