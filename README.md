# 📈 Stock Market Signal Engine with AI & Indicators

This project combines **technical analysis**, **machine learning**, and **sentiment data** to identify **buy/sell opportunities** in the stock market. Built using Python, XGBoost, and LangChain.

---

## 🚀 Features

### ✅ Indicators Used
- **RSI & RSI Divergence (Z-score, Crossover, Rolling Mean)**
- **MACD & MACD Divergence**
- **SuperTrend (Daily & Weekly)**
- **Bollinger Bands**
- **OBV & OBV Divergence**
- **Momentum**
- **Composite Buy/Sell Signal**
- **Reddit Sentiment Analysis (r/stocks)**

### 🧠 Machine Learning
- **Model**: XGBoost Classifier
- **Validation**: TimeSeriesSplit cross-validation
- **Target**: Predict if price will rise next day
- **Features**: 15+ engineered indicators

### 🧩 LangChain Agent
- Use natural language to:
  - Run analysis: `"Analyze SBIN.NS from 2024-01-01 to 2025-05-13"`
  - Fetch indicators: `"SBIN.NS,2024-05-01,RSI"`
  - Get summaries: `"Show latest signal summary for SBIN.NS"`

### 📊 Dashboard (Plotly)
- Candlestick chart
- Buy/Sell signals
- MACD, RSI, Bollinger Bands, Momentum
- OBV and divergence markers

---
