# 📈 DemandSense AI

### Retail Inventory Forecasting & Supply Chain Optimization using Machine Learning

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20WSL-blue)
![Python](https://img.shields.io/badge/python-3.12+-blue)
![Status](https://img.shields.io/badge/status-active-success)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)
![Prophet](https://img.shields.io/badge/Forecasting-Prophet-blue)

---

## 📖 Overview

DemandSense AI is an end-to-end machine learning project that forecasts retail demand using both statistical and deep learning models. The system processes retail sales data through a complete ETL pipeline, trains multiple forecasting models, estimates inventory requirements, and visualizes business impact through an interactive Streamlit dashboard.

The project demonstrates practical machine learning engineering by combining data engineering, time-series forecasting, business analytics, and modern software engineering practices into a modular, production-oriented workflow.

---

## 🎯 Project Goals

This project was built to demonstrate practical implementation of:

- End-to-end Machine Learning pipelines
- Retail demand forecasting
- Time-series Deep Learning
- ETL and feature engineering workflows
- Business KPI analysis
- Interactive analytical dashboards
- Modular software architecture

---

# 🚀 Features

- Synthetic retail sales data generation
- Complete ETL pipeline
- Exploratory Data Analysis (EDA)
- Feature engineering
- Time-series forecasting with Prophet
- Deep Learning forecasting using PyTorch LSTM
- Forecast comparison
- Inventory optimization insights
- Business KPI calculations
- Interactive Streamlit dashboard
- Modular project architecture

---

# 💻 Development Environment

This project was developed and tested on:

- Ubuntu Linux
- Windows Subsystem for Linux (WSL2)
- Python 3.12+

The LSTM training pipeline automatically utilizes CUDA acceleration when a compatible NVIDIA GPU is available, while gracefully falling back to CPU execution otherwise.

---

# 📊 Interactive Dashboard

The Streamlit dashboard allows business users to explore demand forecasts, compare forecasting models, and evaluate inventory decisions through an intuitive interface.

## Dashboard Preview

![Dashboard](assets/dashboard.png)

### Dashboard Features

- Store selection
- Product filtering
- Prophet vs LSTM comparison
- Interactive forecast visualization
- Inventory recommendations
- Estimated overstock reduction
- Estimated stockout prevention
- Business ROI metrics

---

# 🏗️ Project Architecture

```text
Raw Retail Data
       │
       ▼
Data Generation
       │
       ▼
Data Cleaning & ETL
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Feature Engineering
       │
       ├──────────────┐
       ▼              ▼
 Prophet Model     LSTM Model
       │              │
       └──────┬───────┘
              ▼
      Forecast Comparison
              ▼
 Business Metrics & ROI
              ▼
      Streamlit Dashboard
```

---

# 🧠 Machine Learning Pipeline

## 1. Data Engineering

- Synthetic retail sales generation
- Missing value handling
- Date parsing
- Feature engineering
- Data normalization

---

## 2. Exploratory Data Analysis

- Sales trend analysis
- Weekly seasonality
- Rolling averages
- Distribution analysis
- Product-level insights

---

## 3. Statistical Forecasting

Implemented using **Meta Prophet**.

Features include:

- Trend modeling
- Weekly seasonality
- Future demand prediction
- Forecast validation
- Business rule guardrails

---

## 4. Deep Learning Forecasting

Built using **PyTorch LSTM**.

Pipeline includes:

- Sliding window dataset generation
- Sequence modeling
- GPU (CUDA) acceleration
- Multi-epoch training
- Future demand prediction

---

## 🤖 Why Two Forecasting Models?

DemandSense AI compares two fundamentally different forecasting approaches.

### Prophet

- Strong statistical baseline
- Fast training
- Excellent seasonal forecasting
- Interpretable predictions

### PyTorch LSTM

- Learns nonlinear temporal relationships
- Captures complex demand patterns
- Supports GPU acceleration
- Deep learning approach for sequential data

---

## 📈 Business Intelligence Dashboard

Built using **Streamlit**.

The dashboard enables business users to:

- Compare Prophet and LSTM forecasts
- Analyze historical sales trends
- Estimate inventory requirements
- Evaluate business KPIs
- Visualize demand forecasts interactively

---

# ⚡ Performance

The deep learning pipeline supports hardware acceleration using CUDA.

- Automatic GPU detection
- CPU fallback support
- Optimized tensor operations
- Efficient batch processing

---

# ⚙️ Technology Stack

| Category | Technologies |
|-----------|--------------|
| Programming | Python 3.12 |
| Data Engineering | Pandas, NumPy |
| Machine Learning | PyTorch, Prophet, Scikit-Learn |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Development | Linux, WSL2 |
| Version Control | Git, GitHub |

---

# 📂 Repository Structure

```text
demandsense/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── data_prep/
│   ├── features/
│   └── models/
│       ├── train_prophet.py
│       └── train_lstm.py
│
├── dashboard/
│   └── app.py
│
├── assets/
│   └── dashboard.png
│
├── requirements.txt
└── README.md
```

---

# 📈 Example Workflow

```text
Generate Raw Data
        │
        ▼
Clean & Transform Data
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Train Prophet
        │
        ▼
Train LSTM
        │
        ▼
Generate Forecasts
        │
        ▼
Calculate Business KPIs
        │
        ▼
Interactive Dashboard
```

---

# 🛠 Installation

## Clone the repository

```bash
git clone https://github.com/ofcitsjoe/demandsense.git
cd demandsense
```

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Generate synthetic retail data.

```bash
python src/data_prep/generate_raw_data.py
```

Clean the dataset.

```bash
python src/data_prep/clean_data.py
```

Train the Prophet model.

```bash
python src/models/train_prophet.py
```

Train the LSTM model.

```bash
python src/models/train_lstm.py
```

Launch the dashboard.

```bash
streamlit run dashboard/app.py
```

---

# 📈 Example Results

The forecasting pipeline demonstrates:

- Accurate seasonal trend learning
- Multi-model demand forecasting
- Inventory requirement estimation
- Business KPI reporting
- Interactive visualization for decision support

---

# 📚 Learning Objectives

This project demonstrates practical experience with:

- Machine Learning Engineering
- Time-Series Forecasting
- Deep Learning
- LSTM Networks
- ETL Pipelines
- Feature Engineering
- Business Analytics
- Data Visualization
- Streamlit Applications
- Production-oriented Python Architecture

---

# 📌 Future Improvements

- [ ] CSV upload support
- [ ] Real-time inference API
- [ ] Docker containerization
- [ ] PostgreSQL integration
- [ ] Snowflake integration
- [ ] Weather and holiday features
- [ ] Multi-product forecasting
- [ ] Transformer-based forecasting models
- [ ] CI/CD pipeline
- [ ] Cloud deployment (AWS/Azure/GCP)

---

# 📝 Notes

- This project uses synthetically generated retail data for demonstration purposes.
- The repository focuses on showcasing practical machine learning engineering workflows rather than production retail forecasting.

---

# 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

Feel free to fork the repository, open an issue, or submit a pull request.

---

# 📄 License

This project is licensed under the MIT License.

---

## ⭐ If you found this project helpful, consider giving it a star!