import pandas as pd
from prophet import Prophet
import os

def train_prophet_model():
    print("Starting Prophet model training...")

    data_path = "data/processed/cleaned_sales_data.csv"
    df = pd.read_csv(data_path)

    # filter for a specific store and product combination for training
    mask = (df['store_id'] == 'STORE_001') & (df['product_id'] == 'PROD_A')
    model_df = df[mask].copy()

    model_df = model_df[['date', 'sales_volume']].rename(columns={'date': 'ds', 'sales_volume': 'y'})

    model_df['ds'] = pd.to_datetime(model_df['ds'])

    print(f"Data filtered and formatted. Total rows for training: {len(model_df)}")
    print("\n--- PROPHET FORMATTED DATA ---")
    print(model_df.head())

    # instantiate and train the model
    print("\n--- TRAINING PROPHET MODEL ---")
    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model.fit(model_df)

    # create a future timeline (predicting 30 days out)
    # this creates a blank dataframe with our original 60 days + 30 new days
    future = model.make_future_dataframe(periods=30)

    # predict the future
    forecast = model.predict(future)

    print("\n--- FORECAST RESULTS (Next 5 Days) ---")
    # yhat = predicted value, yhat_lower/upper = the confidence interval margins
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[60:65])  # show the first 5 days of the forecast

    # post-processing
    forecast['yhat'] = forecast['yhat'].clip(lower=0)  # ensure no negative predictions
    forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)

    # save the forecast results
    output_dir = "data/processed"
    forecast_path = os.path.join(output_dir, "prophet_forecast.csv")
    forecast.to_csv(forecast_path, index=False)
    print(f"\nSUCCESS: Forecast results saved to {forecast_path}")

if __name__ == "__main__":
    train_prophet_model()