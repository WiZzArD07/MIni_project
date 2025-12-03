import time
import io
import os
import json
import threading
import tempfile
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Optional packages
try:
    from binance import AsyncClient, BinanceSocketManager
    BINANCE_AVAILABLE = True
except Exception:
    BINANCE_AVAILABLE = False

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except Exception:
    PANDAS_TA_AVAILABLE = False

# Legacy Keras loader for Keras 2.x models in Keras 3 runtime
try:
    import keras
    from keras.src.saving import legacy_serialization
    KERAS_LEGACY = True
except Exception:
    KERAS_LEGACY = False

# Streamlit page config
st.set_page_config(page_title="Crypto + Technical Dashboard + Forecast", layout="wide")

# ---------------------------
# Constants / Defaults
# ---------------------------
DEFAULT_MODEL_FILENAME = "Bitcoin_Price_prediction_Model (1).keras"
MODEL_PATH = st.sidebar.text_input("Model filename (uploaded)", value=DEFAULT_MODEL_FILENAME)

# ---------------------------
# Model loading (safe)
# ---------------------------
model = None
model_load_error = None
if KERAS_LEGACY and os.path.exists(MODEL_PATH):
    try:
        model = legacy_serialization.load_model_keras_2(MODEL_PATH)
    except Exception as e:
        model_load_error = str(e)
else:
    if not KERAS_LEGACY:
        model_load_error = "Keras legacy loader not available. Make sure keras==3.0.5 is installed."
    else:
        model_load_error = f"Model file {MODEL_PATH} not found."

# ---------------------------
# Shared state
# ---------------------------
LIVE_PRICES = {}
LIVE_LOCK = threading.Lock()
ASYNC_LOOP = None
ASYNC_TASK = None
ASYNC_RUNNING = False
LAST_RETRAIN_TS = None

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(ttl=60)
def fetch_yf(ticker: str, period: str = "1y", interval: str = "1d"):
    """Fetch historical data from yfinance and return a flattened DataFrame with Date column."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    # Flatten columns if MultiIndex
    df.columns = [c if not isinstance(c, tuple) else c[0] for c in df.columns]
    df = df.reset_index()
    return df

# ---------------------------
# Technical indicators (robust)
# ---------------------------
def add_technical_indicators(df: pd.DataFrame):
    """Return a copy of df with technical indicator columns added. Safe to run across pandas_ta versions."""
    out = df.copy()
    if not PANDAS_TA_AVAILABLE:
        # Provide empty columns to avoid downstream KeyErrors
        for col in ["SMA_20","SMA_50","EMA_20","RSI","MACD","MACD_SIGNAL","BBL","BBM","BBU"]:
            out[col] = pd.NA
        return out

    # Ensure Close exists
    if 'Close' not in out.columns:
        raise ValueError("DataFrame must contain 'Close' column")

    close = out['Close']

    # Moving averages
    try:
        out['SMA_20'] = ta.sma(close, length=20)
        out['SMA_50'] = ta.sma(close, length=50)
        out['EMA_20'] = ta.ema(close, length=20)
    except Exception:
        out['SMA_20'] = out['SMA_50'] = out['EMA_20'] = pd.NA

    # RSI
    try:
        out['RSI'] = ta.rsi(close, length=14)
    except Exception:
        out['RSI'] = pd.NA

    # MACD - dynamic column detection
    try:
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if isinstance(macd, pd.DataFrame):
            macd_cols = list(macd.columns)
            # main MACD line (contains 'MACD' and a number signature)
            macd_main_candidates = [c for c in macd_cols if 'MACD' in c and ('h' not in c.lower() and 'hist' not in c.lower())]
            macd_signal_candidates = [c for c in macd_cols if 'MACDs' in c or 'signal' in c.lower()]
            if macd_main_candidates:
                out['MACD'] = macd[macd_main_candidates[0]]
            else:
                out['MACD'] = pd.NA
            if macd_signal_candidates:
                out['MACD_SIGNAL'] = macd[macd_signal_candidates[0]]
            else:
                out['MACD_SIGNAL'] = pd.NA
        else:
            out['MACD'] = out['MACD_SIGNAL'] = pd.NA
    except Exception:
        out['MACD'] = out['MACD_SIGNAL'] = pd.NA

    # Bollinger Bands - dynamic detection for versions that use different suffix formats
    try:
        bbands = ta.bbands(close, length=20, std=2)
        if isinstance(bbands, pd.DataFrame):
            bcols = list(bbands.columns)
            bbl = next((c for c in bcols if c.startswith('BBL_') or c.startswith('bb_lower') or c.lower().startswith('lowerband')), None)
            bbm = next((c for c in bcols if c.startswith('BBM_') or c.startswith('bb_middle') or c.lower().startswith('middleband')), None)
            bbu = next((c for c in bcols if c.startswith('BBU_') or c.startswith('bb_upper') or c.lower().startswith('upperband')), None)
            out['BBL'] = bbands[bbl] if bbl in bcols else pd.NA
            out['BBM'] = bbands[bbm] if bbm in bcols else pd.NA
            out['BBU'] = bbands[bbu] if bbu in bcols else pd.NA
        else:
            out['BBL'] = out['BBM'] = out['BBU'] = pd.NA
    except Exception:
        out['BBL'] = out['BBM'] = out['BBU'] = pd.NA

    return out

# ---------------------------
# Sequence helpers & forecasting
# ---------------------------
def create_sequences(scaled_data: np.ndarray, base_days: int):
    X = []
    for i in range(base_days, len(scaled_data)):
        X.append(scaled_data[i-base_days:i])
    X = np.array(X).reshape((-1, base_days, 1))
    return X


def iterative_forecast(model, scaler: MinMaxScaler, last_seq: np.ndarray, days: int, base_days: int):
    seq = last_seq.copy()
    preds = []
    for _ in range(days):
        inp = seq.reshape(1, base_days, 1)
        p = model.predict(inp, verbose=0)
        preds.append(p[0][0])
        seq = np.append(seq[1:], p)
    preds = np.array(preds).reshape(-1, 1)
    preds_inv = scaler.inverse_transform(preds)
    return preds_inv.flatten()

# ---------------------------
# Alerts (telegram / email)
# ---------------------------
def send_telegram_message(bot_token: str, chat_id: str, text: str):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": text}, timeout=10)
        return resp.ok
    except Exception:
        return False


def send_email_smtp(smtp_host, smtp_port, smtp_user, smtp_pass, to_email, subject, body):
    import smtplib
    from email.message import EmailMessage
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = smtp_user
        msg['To'] = to_email
        server = smtplib.SMTP(smtp_host, smtp_port, timeout=10)
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print("Email send failed:", e)
        return False

# ---------------------------
# Training (LSTM) helper
# ---------------------------
def train_and_save_model(ticker="BTC-USD", period="3y", base_days=100, epochs=10, batch_size=32, save_path=MODEL_PATH):
    try:
        df = fetch_yf(ticker, period=period, interval="1d")
        series = df[['Close']].dropna()
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(series.values)

        # Create sequences
        X = []
        y = []
        for i in range(base_days, len(scaled)):
            X.append(scaled[i-base_days:i])
            y.append(scaled[i, 0])
        X = np.array(X).reshape(-1, base_days, 1)
        y = np.array(y).reshape(-1, 1)

        # Build model (TensorFlow Keras)
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        model_local = Sequential()
        model_local.add(LSTM(64, return_sequences=True, input_shape=(base_days, 1)))
        model_local.add(Dropout(0.2))
        model_local.add(LSTM(64))
        model_local.add(Dense(32, activation='relu'))
        model_local.add(Dense(1))

        model_local.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        model_local.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

        # Save model
        model_local.save(save_path)
        return True
    except Exception as e:
        print("Training failed:", e)
        return False

# ---------------------------
# Async Binance listener (optional)
# ---------------------------
ASYNC_THREAD = None
ASYNC_STOP = threading.Event()

def _normalize_symbol_to_binance(ticker: str):
    """Convert e.g. BTC-USD or BTC to BTCUSDT safely"""
    t = ticker.upper().replace('-USD','').replace('USD-','').replace('-','')
    if t.endswith('USDT'):
        return t
    return t + 'USDT'


def run_async_binance(symbols):
    global ASYNC_LOOP, ASYNC_TASK, ASYNC_RUNNING
    if not BINANCE_AVAILABLE:
        print("python-binance not installed; cannot start async Binance WS.")
        return

    def _start_loop(loop):
        import asyncio
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def _listen_symbols(symbols):
        try:
            client = await AsyncClient.create()
            bm = BinanceSocketManager(client)
            sockets = [bm.trade_socket(sym.lower()) for sym in symbols]

            async def _listen_one(sym, sock):
                async with sock as s:
                    async for msg in s:
                        try:
                            if 'p' in msg and 's' in msg:
                                price = float(msg['p'])
                                sym_up = msg['s']
                                with LIVE_LOCK:
                                    LIVE_PRICES[sym_up] = price
                        except Exception:
                            continue

            tasks = [asyncio.create_task(_listen_one(sym.upper(), sock)) for sym, sock in zip(symbols, sockets)]
            await asyncio.gather(*tasks)
        except Exception as e:
            print("Async Binance listener error:", e)
        finally:
            try:
                await client.close_connection()
            except Exception:
                pass

    import asyncio
    ASYNC_LOOP = asyncio.new_event_loop()
    t = threading.Thread(target=_start_loop, args=(ASYNC_LOOP,), daemon=True)
    t.start()
    ASYNC_TASK = asyncio.run_coroutine_threadsafe(_listen_symbols(symbols), ASYNC_LOOP)
    ASYNC_RUNNING = True


def stop_async_binance():
    global ASYNC_TASK, ASYNC_LOOP, ASYNC_RUNNING
    try:
        if ASYNC_TASK:
            ASYNC_TASK.cancel()
        if ASYNC_LOOP:
            ASYNC_LOOP.call_soon_threadsafe(ASYNC_LOOP.stop)
        ASYNC_RUNNING = False
    except Exception as e:
        print("Error stopping async binance:", e)

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Crypto Live + Technical Dashboard + Forecasts")

# Top-row controls
col1, col2, col3 = st.columns([2,2,2])
with col1:
    selection = st.multiselect("Choose coins", ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"], default=["BTC-USD","ETH-USD"])
with col2:
    live_interval = st.selectbox("Live chart period", ["1d","5d","1mo","3mo","6mo","1y"], index=0)
    live_interval_small = st.selectbox("Live chart interval", ["1m","2m","5m","15m","30m","60m","1d"], index=6)
with col3:
    start_binance = st.button("Start Live (Binance Async)")
    stop_binance = st.button("Stop Live")
    start_poll = st.button("Start Polling (yfinance)")

# model status
st.sidebar.subheader("Model status")
if model is not None:
    st.sidebar.success("Model loaded ✓")
else:
    st.sidebar.error(f"Model not loaded: {model_load_error}")

# Price Alerts config
st.sidebar.header("Price Alerts (rules)")
alert_ticker = st.sidebar.selectbox("Alert ticker", ["BTC-USD","ETH-USD","SOL-USD","BNB-USD"], index=0)
alert_operator = st.sidebar.selectbox("Operator", ["greater_than","less_than","percent_change_up","percent_change_down"])
alert_value = st.sidebar.number_input("Value (price or percent)", value=50000.0, step=1.0)
telegram_token = st.sidebar.text_input("Telegram Bot Token (optional)", value="")
telegram_chat = st.sidebar.text_input("Telegram Chat ID (optional)", value="")
smtp_host = st.sidebar.text_input("SMTP Host (optional)", value="")
smtp_port = st.sidebar.number_input("SMTP Port", value=587)
smtp_user = st.sidebar.text_input("SMTP User (optional)", value="")
smtp_pass = st.sidebar.text_input("SMTP Password (optional)", value="", type="password")
email_to = st.sidebar.text_input("Notify Email (optional)", value="")

# Auto retrain controls
st.sidebar.header("Retraining")
retrain_enable = st.sidebar.checkbox("Enable daily auto-retrain", value=False)
retrain_interval_hours = st.sidebar.number_input("Retrain interval (hours)", value=24.0, min_value=1.0)
retrain_epochs = st.sidebar.number_input("Retrain epochs", value=4, min_value=1)
retrain_batch = st.sidebar.number_input("Retrain batch_size", value=32, min_value=1)
retrain_period = st.sidebar.selectbox("Retrain data period", ["6mo","1y","2y","3y","5y"], index=3)

# Manual retrain
if st.sidebar.button("Manual retrain now"):
    st.sidebar.info("Retraining... this may take time.")
    params = {"ticker": alert_ticker, "period": retrain_period, "base_days": 100, "epochs": int(retrain_epochs), "batch_size": int(retrain_batch), "save_path": MODEL_PATH}
    ok = train_and_save_model(**params)
    if ok:
        st.sidebar.success("Manual retrain finished. Reloading model...")
        try:
            if KERAS_LEGACY:
                model = legacy_serialization.load_model_keras_2(MODEL_PATH)
            else:
                from tensorflow.keras.models import load_model
                model = load_model(MODEL_PATH)
            st.sidebar.success("Model reloaded.")
        except Exception as e:
            st.sidebar.error(f"Reload failed: {e}")
    else:
        st.sidebar.error("Retrain failed. See logs.")

# Start/stop async binance
if start_binance:
    if BINANCE_AVAILABLE:
        symbols = [_normalize_symbol_to_binance(s) for s in selection]
        try:
            run_async_binance(symbols)
            st.sidebar.success("Started async Binance listener.")
        except Exception as e:
            st.sidebar.error(f"Start failed: {e}")
    else:
        st.sidebar.error("python-binance not installed. Please add python-binance to requirements.")

if stop_binance:
    stop_async_binance()
    st.sidebar.info("Requested stop.")

if start_poll:
    st.sidebar.info("Polling mode: fetching yfinance every refresh.")

# Live display area
st.header("Live Prices")
cols = st.columns(len(selection) if selection else 1)
for i, ticker in enumerate(selection):
    bin_sym = _normalize_symbol_to_binance(ticker)
    price = None
    with LIVE_LOCK:
        price = LIVE_PRICES.get(bin_sym)
    if price is None:
        try:
            df = fetch_yf(ticker, period="5d", interval="1d")
            price = float(df['Close'].iloc[-1])
        except Exception:
            price = None
    with cols[i]:
        if price:
            st.metric(ticker, f"${price:,.2f}")
        else:
            st.metric(ticker, "N/A")

# Technical indicator dashboard & historical chart for main coin (first selection or BTC-USD)
main_coin = selection[0] if selection else "BTC-USD"
st.header(f"Technical Dashboard — {main_coin}")
hist_df = fetch_yf(main_coin, period="1y", interval="1d")
if 'Date' in hist_df.columns:
    hist_df['Date'] = pd.to_datetime(hist_df['Date'])
ind_df = add_technical_indicators(hist_df)

# Show last 10 rows
st.subheader("Recent data + indicators")
st.dataframe(ind_df.tail(10))

# Plot Close + MA/EMA + Bollinger (only columns that exist)
plot_cols = [c for c in ['Close','SMA_20','SMA_50','EMA_20','BBL','BBM','BBU'] if c in ind_df.columns]
fig_df = ind_df.set_index('Date')[plot_cols].dropna(how='all')
if not fig_df.empty:
    st.line_chart(fig_df)

# RSI
if 'RSI' in ind_df.columns and ind_df['RSI'].notna().any():
    st.subheader("RSI (14)")
    st.line_chart(ind_df.set_index('Date')['RSI'].dropna())

# MACD
if 'MACD' in ind_df.columns and 'MACD_SIGNAL' in ind_df.columns and ind_df[['MACD','MACD_SIGNAL']].notna().any().any():
    st.subheader("MACD")
    st.line_chart(ind_df.set_index('Date')[['MACD','MACD_SIGNAL']].dropna())

# ---------------------------
# Forecasting (multi-horizon)
# ---------------------------
st.header("Forecasts (Model-based)")
full = fetch_yf(main_coin, period="5y", interval="1d")
if 'Date' in full.columns:
    full['Date'] = pd.to_datetime(full['Date'])
st.subheader("Recent historical")
st.write(full.tail(8))

if model is None:
    st.warning("Model not loaded; forecasts are unavailable.")
else:
    series = full[['Close']].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values)
    base_days = 100
    if len(scaled) < base_days + 1:
        st.error("Not enough data for forecasting (increase historical period).")
    else:
        last_seq = scaled[-base_days:].reshape(-1)
        horizons = [7,30,90]
        all_dfs = {}
        for h in horizons:
            preds = iterative_forecast(model, scaler, last_seq, h, base_days)
            start_date = full['Date'].iloc[-1] + pd.Timedelta(days=1)
            dates = pd.date_range(start_date, periods=h, freq='D')
            df_preds = pd.DataFrame({"Date": dates, "Predicted Price": preds})
            all_dfs[h] = df_preds
            st.write(f"### {h}-day forecast")
            st.dataframe(df_preds)
            st.line_chart(df_preds.set_index('Date')['Predicted Price'])

        # CSV download
        combined = []
        for h, dfh in all_dfs.items():
            d = dfh.copy()
            d['Horizon'] = h
            combined.append(d)
        outdf = pd.concat(combined, ignore_index=True)
        buf = io.StringIO()
        outdf.to_csv(buf, index=False)
        st.download_button("Download forecasts CSV", buf.getvalue().encode('utf-8'),
                           file_name=f"forecasts_{main_coin}.csv", mime="text/csv")

# ---------------------------
# Alerts & Notifications (UI)
# ---------------------------
st.header("Alerts & Notifications")

def check_rule_and_notify(ticker, op, val):
    bin_sym = _normalize_symbol_to_binance(ticker)
    with LIVE_LOCK:
        price = LIVE_PRICES.get(bin_sym)
    if price is None:
        try:
            df = fetch_yf(ticker, period="5d", interval="1d")
            price = float(df['Close'].iloc[-1])
        except Exception:
            return False, "price_unavailable"

    triggered = False
    msg = ""
    if op == "greater_than":
        if price > val:
            triggered = True
            msg = f"{ticker} is above {val}: {price}"
    elif op == "less_than":
        if price < val:
            triggered = True
            msg = f"{ticker} is below {val}: {price}"
    elif op in ("percent_change_up", "percent_change_down"):
        try:
            df = fetch_yf(ticker, period="2d", interval="1d")
            prev = float(df['Close'].iloc[-2])
            pct = (price - prev) / prev * 100.0
            if op == "percent_change_up" and pct >= val:
                triggered = True
                msg = f"{ticker} rose {pct:.2f}% >= {val}% (price {price})"
            if op == "percent_change_down" and pct <= -val:
                triggered = True
                msg = f"{ticker} dropped {pct:.2f}% <= -{val}% (price {price})"
        except Exception:
            pass

    if triggered:
        if telegram_token and telegram_chat:
            send_telegram_message(telegram_token, telegram_chat, msg)
        if smtp_host and smtp_user and smtp_pass and email_to:
            send_email_smtp(smtp_host, smtp_port, smtp_user, smtp_pass, email_to, f"Alert: {ticker}", msg)
    return triggered, msg

if st.button("Check alert now"):
    res, m = check_rule_and_notify(alert_ticker, alert_operator, alert_value)
    if res:
        st.success(f"Alert triggered: {m}")
    else:
        st.info(f"No alert (status: {m})")

# ---------------------------
# Colab notebook generation
# ---------------------------
st.header("Generate Google Colab LSTM training notebook")
if st.button("Generate & Download Colab notebook"):
    notebook = {
      "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# LSTM training notebook\n", "This notebook trains a simple LSTM on historical Close prices (yfinance) and saves a `.keras` model compatible with Keras 3."]},
        {"cell_type": "code", "metadata": {}, "source": ["!pip install -q yfinance pandas numpy scikit-learn tensorflow keras\n"], "outputs": []},
        {"cell_type": "code", "metadata": {}, "source": [
            "import yfinance as yf\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "from sklearn.preprocessing import MinMaxScaler\n",
            "from tensorflow.keras.models import Sequential\n",
            "from tensorflow.keras.layers import LSTM, Dense, Dropout\n",
            "from tensorflow.keras.optimizers import Adam\n",
            "\n",
            "TICKER = 'BTC-USD'\n",
            "PERIOD = '3y'\n",
            "BASE_DAYS = 100\n",
            "EPOCHS = 10\n",
            "BATCH = 32\n",
            "\n",
            "df = yf.download(TICKER, period=PERIOD, interval='1d')\n",
            "df = df.reset_index()\n",
            "series = df[['Close']].dropna()\n",
            "scaler = MinMaxScaler()\n",
            "scaled = scaler.fit_transform(series.values)\n",
            "\n",
            "X = []\n",
            "y = []\n",
            "for i in range(BASE_DAYS, len(scaled)):\n",
            "    X.append(scaled[i-BASE_DAYS:i])\n",
            "    y.append(scaled[i,0])\n",
            "X = np.array(X).reshape(-1, BASE_DAYS, 1)\n",
            "y = np.array(y).reshape(-1,1)\n",
            "\n",
            "model = Sequential()\n",
            "model.add(LSTM(64, return_sequences=True, input_shape=(BASE_DAYS,1)))\n",
            "model.add(Dropout(0.2))\n",
            "model.add(LSTM(64))\n",
            "model.add(Dense(32, activation='relu'))\n",
            "model.add(Dense(1))\n",
            "\n",
            "model.compile(optimizer=Adam(0.001), loss='mse')\n",
            "model.fit(X, y, epochs=EPOCHS, batch_size=BATCH)\n",
            "\n",
            "model.save('trained_model.keras')\n",
            "print('Saved trained_model.keras')\n"
        ], "outputs": []}
      ],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4,
      "nbformat_minor": 5
    }
    notebook_str = json.dumps(notebook)
    b = notebook_str.encode('utf-8')
    st.download_button("Download Colab notebook (.ipynb)", data=b, file_name="lstm_train_colab.ipynb", mime="application/json")

# ---------------------------
# Auto-retrain scheduler
# ---------------------------
if retrain_enable:
    if 'retrain_thread' not in st.session_state:
        st.session_state['retrain_stop'] = threading.Event()
        params = {"ticker": alert_ticker, "period": retrain_period, "base_days": 100, "epochs": int(retrain_epochs), "batch_size": int(retrain_batch), "save_path": MODEL_PATH}
        t = threading.Thread(target=retrain_model_worker, args=(retrain_interval_hours, params, st.session_state['retrain_stop']), daemon=True)
        t.start()
        st.session_state['retrain_thread'] = t
        st.sidebar.success("Auto-retrain started in background.")
    else:
        st.sidebar.info("Auto-retrain already running.")
else:
    if 'retrain_stop' in st.session_state:
        st.session_state['retrain_stop'].set()
        st.sidebar.info("Auto-retrain stopped.")
        st.session_state.pop('retrain_thread', None)
        st.session_state.pop('retrain_stop', None)

# ---------------------------
# UI auto-refresh
# ---------------------------
if st.sidebar.checkbox("Auto refresh main UI (small interval)", value=True):
    interval = st.sidebar.number_input("UI refresh interval seconds", value=2, min_value=1, max_value=60)
    if 'last_rerun' not in st.session_state:
        st.session_state['last_rerun'] = time.time()
    if time.time() - st.session_state['last_rerun'] > interval:
        st.session_state['last_rerun'] = time.time()
        st.experimental_rerun()
