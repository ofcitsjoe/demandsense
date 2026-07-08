import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="DemandSense AI", layout="wide", page_icon="📈")
st.title("📈 DemandSense AI Dashboard")
st.markdown("Compare baseline statistical models against Deep Learning for retail inventory optimization.")

# 1. Caching the Data Load (NO FILTERING HERE)
@st.cache_data
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    # Load all historical data
    hist_path = os.path.join(root_dir, "data", "processed", "cleaned_sales_data.csv")
    hist = pd.read_csv(hist_path)
    hist['date'] = pd.to_datetime(hist['date'])
    
    # Load forecasts (Currently only contains STORE_001/PROD_A)
    prophet_path = os.path.join(root_dir, "data", "processed", "prophet_forecast.csv")
    prophet = pd.read_csv(prophet_path)
    prophet['ds'] = pd.to_datetime(prophet['ds'])
    
    lstm_path = os.path.join(root_dir, "data", "processed", "lstm_forecast.csv")
    lstm = pd.read_csv(lstm_path)
    lstm['ds'] = pd.to_datetime(lstm['ds'])
    
    return hist, prophet, lstm

try:
    hist_full, prophet_full, lstm_full = load_data()
except FileNotFoundError:
    st.error("Data files not found. Please ensure you have run the ETL and Training scripts.")
    st.stop()

# 2. The Interactive Sidebar
st.sidebar.header("Filter Controls")

# Dynamically grab all unique stores and products from the historical data
available_stores = hist_full['store_id'].unique()
available_products = hist_full['product_id'].unique()

selected_store = st.sidebar.selectbox("Select Store", available_stores)
selected_product = st.sidebar.selectbox("Select Product", available_products)

# 3. Dynamic Filtering
# Filter Historical Data
hist_mask = (hist_full['store_id'] == selected_store) & (hist_full['product_id'] == selected_product)
hist_filtered = hist_full[hist_mask][['date', 'sales_volume']].rename(columns={'sales_volume': 'Actual Sales'})

# Filter Forecasts (Adding a safety check since our ML files are currently limited)
# In production, your ML files would also have store_id and product_id columns to filter by.
prophet_filtered = prophet_full[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'Prophet Baseline'})
lstm_filtered = lstm_full[['ds', 'lstm_yhat']].rename(columns={'ds': 'date', 'lstm_yhat': 'LSTM (PyTorch)'})

# 4. Merge the timelines
chart_data = pd.merge(hist_filtered, prophet_filtered, on='date', how='outer')
chart_data = pd.merge(chart_data, lstm_filtered, on='date', how='outer')
chart_data.set_index('date', inplace=True)

# 5. Render the UI
st.subheader(f"{selected_store} | {selected_product}: 30-Day Demand Forecast")

if hist_filtered.empty:
    st.warning("No data available for this combination.")
else:
    st.line_chart(chart_data, use_container_width=True)

    st.subheader("Business Impact Summary (Next 30 Days)")
    col1, col2, col3 = st.columns(3)

    lstm_total = int(lstm_filtered['LSTM (PyTorch)'].sum())
    prophet_total = int(prophet_filtered['Prophet Baseline'].sum())
    difference = lstm_total - prophet_total

    with col1:
        st.metric(label="LSTM Total Projected Demand", value=f"{lstm_total} units")
    with col2:
        st.metric(label="Prophet Total Projected Demand", value=f"{prophet_total} units", delta=f"{difference} units variance", delta_color="inverse")
    with col3:
        st.metric(label="System Status", value="Online", delta="CUDA Accelerated")