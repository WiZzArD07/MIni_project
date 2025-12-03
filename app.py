# app.py
import time
import io
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Crypto Live + Forecast", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Load model (trained on BTC)
# -------------------------
MODEL_PATH = "Bitcoin_Price_prediction_Model.keras"
model = load_model(MODEL_PATH)

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
    """Return single-column DataFrame of price_col and a fitted scaler on train portion (we'll fit on last year only)."""
    if price_col not in df.columns:
        raise KeyError(f"{price_col} not in dataframe columns: {df.columns.tolist()}")
    series = df[[price_col]].copy()
    return series

def create_sequences(scaled_data: np.ndarray, base_days: int):
    X = []
    for i in range(base_days, len(scaled_data)):
        X.append(scaled_data[i-base_days:i])
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X

def iterative_forecast(model, scaler: MinMaxScaler, last_sequence: np.ndarray, days: int, base_days: int):
    """last_sequence: 1D array of scaled values length==base_days"""
    preds = []
    seq = last_sequence.copy()
    for _ in range(days):
        input_seq = seq.reshape(1, base_days, 1)
        p = model.predict(input_seq, verbose=0)
        # p is scaled prediction (because model was trained on scaled data), append and slide
        preds.append(p[0][0])
        seq = np.append(seq[1:], p)
    preds = np.array(preds).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds)
    return preds_inv.flatten()

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.title("Controls")

theme_choice = st.sidebar.selectbox("Theme", ["Light", "Dark"])
# Apply simple dark mode CSS
if theme_choice == "Dark":
    st.markdown(
        """
        <style>
        .css-1d391kg {background-color: #0e1117;} 
        .css-1v3fvcr {color: #e6eef6;}
        .stButton>button {background-color: #1f2937; color: #e6eef6;}
        .streamlit-expanderHeader {color: #e6eef6;}
        </style>
        """,
        unsafe_allow_html=True,
    )

auto_refresh = st.sidebar.checkbox("Auto-refresh every 10 seconds", value=True)
refresh_interval = st.sidebar.number_input("Refresh interval (seconds)", min_value=5, max_value=300, value=10, step=1)
if auto_refresh:
    # initialize last_refresh timestamp
    if "last_refresh_ts" not in st.session_state:
        st.session_state["last_refresh_ts"] = time.time()
    # rerun when interval passes
    if time.time() - st.session_state["last_refresh_ts"] > float(refresh_interval):
        st.session_state["last_refresh_ts"] = time.time()
        st.experimental_rerun()

# Select tickers to compare (default includes requested coins)
available_tickers = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Solana (SOL-USD)": "SOL-USD",
    "BNB (BNB-USD)": "BNB-USD"
}
selected_compare = st.sidebar.multiselect("Select coins to compare (live chart)", list(available_tickers.keys()),
                                          default=list(available_tickers.keys()))
selected_tickers = [available_tickers[name] for name in selected_compare]

# Main ticker for forecasting (default BTC)
main_ticker = st.sidebar.selectbox("Main ticker for forecasting", list(available_tickers.values()), index=0)
base_days = st.sidebar.number_input("Base days (lookback for model)", min_value=10, max_value=365, value=100, step=1)

# Forecast horizons
horizons = st.sidebar.multiselect("Forecast horizons (days)", [7, 30, 90], default=[7, 30, 90])

# Download filename prefix
download_prefix = st.sidebar.text_input("Download filename prefix", value="predictions")

# Live chart options
live_period = st.sidebar.selectbox("Live chart period", ["1d", "5d", "1mo", "3mo", "6mo", "1y"], index=0)
live_interval = st.sidebar.selectbox("Live chart interval", ["1m", "2m", "5m", "15m", "30m", "60m", "1d"], index=5)

# -------------------------
# Layout - top: title + subtitle
# -------------------------
st.title("📈 Crypto Live Tracker + Forecast")
st.markdown("Real-time price charts, multi-horizon forecasts (7/30/90 days), CSV export, theme toggle, and coin comparison.")

# -------------------------
# Fetch and show data for comparison tickers
# -------------------------
st.subheader("Live Price Comparison")
col_live = st.columns(1)
with col_live[0]:
    # fetch each ticker as small timeframe for near-real-time display
    live_dfs = {}
    for t in selected_tickers:
        try:
            df_live = fetch_data(t, period=live_period, interval=live_interval)
            # Ensure 'Date' column exists and is datetime
            if 'Date' in df_live.columns:
                df_live['Date'] = pd.to_datetime(df_live['Date'])
                df_live = df_live.set_index('Date')
            live_dfs[t] = df_live
        except Exception as e:
            st.warning(f"Could not fetch {t}: {e}")

    # Combine close prices for comparison
    if live_dfs:
        combined = pd.DataFrame()
        for t, df in live_dfs.items():
            # prefer 'Close' column
            colname = f"{t}"
            if 'Close' in df.columns:
                combined[colname] = df['Close']
            elif 'Adj Close' in df.columns:
                combined[colname] = df['Adj Close']
        if not combined.empty:
            st.line_chart(combined)
        else:
            st.write("No close price columns available to display comparison.")

# -------------------------
# Main ticker section: historical table + forecasts
# -------------------------
st.markdown("---")
st.header(f"Forecast & Analysis — {main_ticker}")

# Fetch 3+ years of data for forecasting UI
full_range = fetch_data(main_ticker, period="5y", interval="1d")
st.subheader("Historical Price Data (latest)")
st.write(full_range.tail(10))

# Chart of Close price (historical)
st.subheader("Close Price (historical)")
if 'Date' in full_range.columns:
    full_range['Date'] = pd.to_datetime(full_range['Date'])
    hist_chart_df = full_range[['Date', 'Close']].set_index('Date')
    st.line_chart(hist_chart_df)
else:
    st.write("Date column missing — cannot plot historical chart.")

# -------------------------
# Prepare data for forecasting with model
# -------------------------
st.subheader("Multi-horizon Forecasts (using loaded model)")
series = prepare_series_for_model(full_range, price_col="Close")

# Fit scaler on the full series (the same approach we used earlier)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_full = scaler.fit_transform(series.values)

# Ensure we have enough points
if len(scaled_full) < base_days + 1:
    st.error(f"Not enough data for base_days={base_days}. Need at least base_days+1 data points.")
else:
    # Create X for evaluation (prediction over test tail)
    X_eval = create_sequences(scaled_full, base_days)
    # Use last sequence for iterative forecasting
    last_seq = scaled_full[-base_days:].reshape(-1)  # 1D

    # Compute predictions for requested horizons
    all_preds = {}
    for h in horizons:
        preds = iterative_forecast(model, scaler, last_seq, days=h, base_days=base_days)
        # Build dates for future horizon (business days approx using daily frequency)
        last_date = pd.to_datetime(full_range['Date'].iloc[-1])
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=h, freq='D')
        df_preds = pd.DataFrame({"Date": future_dates, "Predicted Price": preds})
        all_preds[h] = df_preds
        st.markdown(f"**{h}-day Forecast**")
        st.write(df_preds)
        st.line_chart(df_preds.set_index("Date")["Predicted Price"])

    # Create a combined CSV for download (concatenate horizons with a column indicating horizon)
    combined_download = []
    for h, dfh in all_preds.items():
        d = dfh.copy()
        d["Horizon (days)"] = h
        d["Ticker"] = main_ticker
        combined_download.append(d)
    if combined_download:
        download_df = pd.concat(combined_download, ignore_index=True)
        # CSV buffer
        csv_buffer = io.StringIO()
        download_df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        # Download button
        st.download_button(
            label="Download all forecasts as CSV",
            data=csv_bytes,
            file_name=f"{download_prefix}_{main_ticker}_forecasts.csv",
            mime="text/csv",
        )

# -------------------------
# Optional: Quick model check plot (predicted vs actual for last segment)
# -------------------------
st.markdown("---")
st.subheader("Model check — Predicted vs Actual (last available segment)")

# Build an evaluation comparison if possible (predict on last sequences and compare with actual next points)
if len(scaled_full) >= base_days + 1:
    # generate predictions for each step in the tail to compare
    # We'll do a rolling-window prediction on the last len(series)-base_days elements
    eval_X = create_sequences(scaled_full, base_days)
    # Predict (this predicts starting from base_days -> end)
    try:
        preds_scaled = model.predict(eval_X, verbose=0)
        preds_inv = scaler.inverse_transform(preds_scaled)
        actuals_inv = scaler.inverse_transform(scaled_full[base_days:].reshape(-1, 1))
        compare_df = pd.DataFrame({
            "Predicted": preds_inv.flatten(),
            "Actual": actuals_inv.flatten()
        }, index=full_range['Date'].iloc[base_days:].reset_index(drop=True))
        st.write(compare_df.tail(30))
        st.line_chart(compare_df.tail(200))
    except Exception as e:
        st.warning(f"Could not run evaluation predict: {e}")

# -------------------------
# Footer / extra actions
# -------------------------
st.markdown("---")
st.write("Notes:")
st.write(
    "- The model you supplied was trained for Bitcoin. Using it for ETH/SOL/BNB forecasting may produce unreliable results.\n"
    "- 'Live' charts use yfinance intervals; the actual refresh frequency depends on the exchange and yfinance limits.\n"
    "- Auto-refresh is implemented via `st.experimental_rerun()` — toggle it off if you prefer manual refresh."
)
