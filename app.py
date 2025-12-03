# app.py
import time
import io
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# ⭐ FIX: Use legacy loader for old Keras model
import keras
from keras.src.saving import legacy_serialization

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Crypto Live + Forecast", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Load model (trained on BTC)
# -------------------------
MODEL_PATH = "Bitcoin_Price_prediction_Model (1).keras"

# ⭐ FIX: Load Keras 2.x model inside Keras 3 environment
model = legacy_serialization.load_model_keras_2(MODEL_PATH)

# -------------------------
# Helper functions
# -------------------------
@st.cache_data(ttl=30)
def fetch_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch data via yfinance and flatten MultiIndex columns if present."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    # Flatten MultiIndex columns if any
    df.columns = [col if not isinstance(col, tuple) else col[0] for col in df.columns]
    if "Date" not in df.reset_index().columns:
        df = df.reset_index()
    else:
        df = df.reset_index()
    return df

def prepare_series_for_model(df: pd.DataFrame, price_col: str = "Close"):
    if price_col not in df.columns:
        raise KeyError(f"{price_col} not in dataframe columns: {df.columns.tolist()}")
    return df[[price_col]].copy()

def create_sequences(scaled_data: np.ndarray, base_days: int):
    X = []
    for i in range(base_days, len(scaled_data)):
        X.append(scaled_data[i-base_days:i])
    X = np.array(X).reshape((-1, base_days, 1))
    return X

def iterative_forecast(model, scaler: MinMaxScaler, last_sequence: np.ndarray, days: int, base_days: int):
    preds = []
    seq = last_sequence.copy()
    for _ in range(days):
        inp = seq.reshape(1, base_days, 1)
        p = model.predict(inp, verbose=0)
        preds.append(p[0][0])
        seq = np.append(seq[1:], p)
    preds = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds).flatten()

# -------------------------
# Sidebar
# -------------------------
st.sidebar.title("Controls")

theme_choice = st.sidebar.selectbox("Theme", ["Light", "Dark"])
if theme_choice == "Dark":
    st.markdown("""
        <style>
        .css-1d391kg {background-color: #0e1117;}
        .css-1v3fvcr {color: #e6eef6;}
        .stButton>button {background-color: #1f2937; color: #e6eef6;}
        .streamlit-expanderHeader {color: #e6eef6;}
        </style>
    """, unsafe_allow_html=True)

auto_refresh = st.sidebar.checkbox("Auto-refresh every 10 seconds", value=True)
refresh_interval = st.sidebar.number_input("Refresh interval (seconds)", min_value=5, max_value=300, value=10)

if auto_refresh:
    if "last_refresh_ts" not in st.session_state:
        st.session_state["last_refresh_ts"] = time.time()
    if time.time() - st.session_state["last_refresh_ts"] > float(refresh_interval):
        st.session_state["last_refresh_ts"] = time.time()
        st.experimental_rerun()

available_tickers = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Solana (SOL-USD)": "SOL-USD",
    "BNB (BNB-USD)": "BNB-USD"
}

selected_compare = st.sidebar.multiselect("Select coins to compare", list(available_tickers.keys()),
                                          default=list(available_tickers.keys()))
selected_tickers = [available_tickers[n] for n in selected_compare]

main_ticker = st.sidebar.selectbox("Main ticker for forecasting", list(available_tickers.values()))
base_days = st.sidebar.number_input("Base days", min_value=10, max_value=365, value=100)
horizons = st.sidebar.multiselect("Forecast horizons", [7, 30, 90], default=[7, 30, 90])
download_prefix = st.sidebar.text_input("Download filename prefix", "predictions")

live_period = st.sidebar.selectbox("Live chart period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"])
live_interval = st.sidebar.selectbox("Live chart interval", ["1m", "2m", "5m", "15m", "30m", "60m", "1d"])

# -------------------------
# Live Price Comparison
# -------------------------
st.title("📈 Crypto Live Tracker + Forecast")
st.subheader("Live Price Comparison")

dfs_live = {}
for t in selected_tickers:
    try:
        df = fetch_data(t, period=live_period, interval=live_interval)
        df["Date"] = pd.to_datetime(df["Date"])
        dfs_live[t] = df.set_index("Date")
    except:
        pass

if dfs_live:
    combined = pd.DataFrame()
    for t, df in dfs_live.items():
        if "Close" in df.columns:
            combined[t] = df["Close"]
    st.line_chart(combined)

st.markdown("---")
st.header(f"Forecast & Analysis — {main_ticker}")

# -------------------------
# Historical Data
# -------------------------
full_range = fetch_data(main_ticker, period="5y", interval="1d")
full_range["Date"] = pd.to_datetime(full_range["Date"])

st.subheader("Historical Price Data")
st.write(full_range.tail(10))

hist_df = full_range[["Date", "Close"]].set_index("Date")
st.line_chart(hist_df)

# -------------------------
# Forecasting
# -------------------------
st.subheader("Multi-horizon Forecasts")

series = prepare_series_for_model(full_range)
scaler = MinMaxScaler()
scaled_full = scaler.fit_transform(series.values)

if len(scaled_full) < base_days + 1:
    st.error("Not enough data for forecasting.")
else:
    last_seq = scaled_full[-base_days:].reshape(-1)
    all_preds = {}

    for h in horizons:
        preds = iterative_forecast(model, scaler, last_seq, h, base_days)
        last_date = full_range["Date"].iloc[-1]
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=h)
        df = pd.DataFrame({"Date": future_dates, "Predicted Price": preds})
        all_preds[h] = df

        st.write(f"### {h}-day Forecast")
        st.write(df)
        st.line_chart(df.set_index("Date"))

    # CSV download
    combined = []
    for h, df in all_preds.items():
        d = df.copy()
        d["Horizon"] = h
        combined.append(d)

    out = pd.concat(combined)
    buf = io.StringIO()
    out.to_csv(buf, index=False)

    st.download_button("Download all forecasts", buf.getvalue().encode(),
                       file_name=f"{download_prefix}_{main_ticker}.csv",
                       mime="text/csv")

# -------------------------
# Model Check
# -------------------------
st.markdown("---")
st.subheader("Model check — Predicted vs Actual")

if len(scaled_full) >= base_days + 1:
    eval_X = create_sequences(scaled_full, base_days)
    preds_scaled = model.predict(eval_X, verbose=0)
    preds = scaler.inverse_transform(preds_scaled)
    actual = scaler.inverse_transform(scaled_full[base_days:].reshape(-1, 1))

    compare = pd.DataFrame({
        "Predicted": preds.flatten(),
        "Actual": actual.flatten()
    }, index=full_range["Date"].iloc[base_days:].reset_index(drop=True))

    st.write(compare.tail(30))
    st.line_chart(compare.tail(200))

# Footer
st.markdown("---")
st.write("Note: Model was originally trained for Bitcoin only. Using it on other coins may be inaccurate.")
