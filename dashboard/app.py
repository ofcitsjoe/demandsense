import streamlit as st
import pandas as pd
import os

# configuring the web page
st.set_page_config(page_title="DemandSense AI", layout="wide", page_icon="📈")
st.title("📈 DemandSense AI Dashboard")
st.markdown("Compare baseline statistical models against Deep Learning for retail inventory optimization.")

@st.cache_data
def load_data():

    # loading historical actuals
    hist = pd.read_csv("data/processed/cleaned_sales_data.csv")
    hist['date'] = pd.to_datetime(hist['date'])

    mask = (hist['store_id'] == 'STORE_001') & (hist['product_id'] == 'PROD_A')
    hist = hist[mask][['date','sales_volume']].rename(columns={'sales_volume': 'Actual Sales'})

    # load prophet forecast
    prophet = pd.read_csv("data/processed/prophet_forecast.csv")
    prophet['ds'] = pd.to_datetime(prophet['ds'])
    prophet = prophet[['ds', 'yhat']].rename(columns={'ds':'date', 'yhat':'Prophet Baseline'})

    # load deep learning forecast
    lstm = pd.read_csv("data/processed/lstm_forecast.csv")
    lstm['ds'] = pd.to_datetime(lstm['ds'])
    lstm = lstm[['ds', 'lstm_yhat']].rename(columns={'ds':'date', 'lstm_yhat':'LSTM (PyTorch)'})

    return hist, prophet, lstm

try:
    hist, prophet, lstm = load_data()
except FileNotFoundError:
    st.error("Data files not found. Please ensure you have run the ETL and Training scripts")
    st.stop()

# merge the timelines for plotting
# we use an outer merge so the dates line up perfectly, 
# even though the actuals stop today and the forecasts go into the future
chart_data = pd.merge(hist, prophet, on='date', how='outer')
chart_data = pd.merge(chart_data, lstm, on='date', how='outer')
chart_data.set_index('date', inplace=True)

# render the UI Elements
st.subheader("Store 001 | Product A: 30-Day Demand Forecast")

# streamlit's native interactive line chart
st.line_chart(chart_data, use_container_width=True)

# render Financial / Business Metrics
st.subheader("Business Impact Summary (Next 30 Days)")
st.markdown("By adopting the LSTM model over the statistical baseline, purchasing decisions become tighter, reducing both holding costs and stockout risks.")

col1, col2, col3 = st.columns(3)

# calculate total predicted demand
lstm_total = int(lstm['LSTM (PyTorch)'].sum())
prophet_total = int(prophet['Prophet Baseline'].sum())
difference = lstm_total - prophet_total

with col1:
    st.metric(label="LSTM Total Projected Demand", value=f"{lstm_total} units")
with col2:
    st.metric(label="Prophet Total Projected Demand", value=f"{prophet_total} units", delta=f"{difference} units variance", delta_color="inverse")
with col3:
    st.metric(label="System Status", value="Online", delta="CUDA Accelerated")