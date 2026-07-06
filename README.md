# DemandSense Forecaster 📈

DemandSense Forecaster is an end-to-end predictive analytics pipeline designed to optimize retail supply chains. By ingesting historical multi-store sales data, this project builds a time-series machine learning model to predict future inventory demand, minimizing stockouts and reducing overstock costs.

## 🚀 Features (In Progress)
- [x] **Data Engineering:** Automated ETL pipeline using Pandas to clean, transform, and impute missing retail data.
- [ ] **Exploratory Data Analysis (EDA):** Visualizing seasonality, trends, and anomalies.
- [ ] **Baseline Forecasting:** Time-series prediction using Meta's Prophet.
- [ ] **Deep Learning:** Advanced non-linear demand forecasting using LSTM neural networks.
- [ ] **Business Dashboard:** Streamlit UI for visualizing predictions and calculating ROI.

## 🏗️ Architecture
* **Language:** Python 3.10
* **Data Manipulation:** Pandas, NumPy
* **Environment:** Miniconda, WSL

## 📂 Project Structure
```text
demandsense-forecaster/
├── data/
│   ├── raw/           # Immutable raw data
│   └── processed/     # Cleaned, model-ready data
├── notebooks/         # EDA and prototyping
├── src/
│   ├── data_prep/     # ETL scripts
│   ├── models/        # Forecasting logic
│   └── utils/         # Helper functions
├── dashboard/         # Streamlit application
└── tests/             # Unit testing