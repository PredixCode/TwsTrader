import os
import time
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from tws_wrapper.stock import TwsStock, TwsConnection



def gpu_check():
    print("--- Verifying GPU Availability ---")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"✅ Found {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(f"❌ Error during GPU setup: {e}")
    else:
        print("❌ No GPU detected by TensorFlow. Running on CPU.")
    print("------------------------------------")
    time.sleep(1)

def set_random_seed(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---- IB/TWS defaults via env (override as needed) ----
IB_HOST = os.getenv("IB_HOST", "127.0.0.1")
IB_PORT = int(os.getenv("IB_PORT", "7497"))         # 7497 paper / 7496 live (default IBGW)
IB_CLIENT_ID = int(os.getenv("IB_CLIENT_ID", "1"))
IB_CURRENCY = os.getenv("IB_CURRENCY", "EUR")

def _build_tws_stock(ticker: str) -> TwsStock:
    """
    Helper to construct a TwsStock with a live TwsConnection.
    Adjust env vars above or pass your own connection if integrating elsewhere.
    """
    conn = TwsConnection(host=IB_HOST, port=IB_PORT, client_id=IB_CLIENT_ID)
    stock = TwsStock(
        connection=conn,
        symbol=ticker,
        currency=IB_CURRENCY,
    )
    return stock

def fetch_price_df(ticker: str, barSize: str) -> pd.DataFrame:
    """
    Fetch OHLCV via Interactive Brokers (IB) using TwsStock.
    Mirrors the previous YFinance behavior:
      - get merged 5m/2m/1m with period='max'
      - persist last fetch to CSV
      - return only ['Open','High','Low','Close','Volume'] with dropna()
    """
    stock = _build_tws_stock(ticker)
    df = stock.get_historical_data(period="max", interval=barSize)
    if df is None or df.empty:
        return pd.DataFrame(columns=['Open','High','Low','Close','Volume'])

    # Save last fetch as CSV (same behavior as before)
    stock.last_fetch_to_csv(df)

    cols_needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[cols_needed].dropna()
    return df


def add_mid_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Mid' not in df.columns:
        df['Mid'] = (df['High'] + df['Low']) / 2.0
    return df


def create_price_sequences(df: pd.DataFrame, sequence_length: int, feature_cols: list[str], target_cols: list[str], stride: int = 1):
    df = add_mid_column(df)
    df = add_time_features(df)  # <--- add this

    feat = df[feature_cols].values
    tgt = df[target_cols].shift(-1).dropna().values
    feat = feat[:len(tgt)]

    X, y = [], []
    for i in range(sequence_length, len(feat) + 1, stride):
        X.append(feat[i - sequence_length:i])
        y.append(tgt[i - 1])
    return np.array(X), np.array(y)

def add_time_features(df: pd.DataFrame, intraday_threshold_hours: int = 23) -> pd.DataFrame:
    """
    Adds cyclical time features based on DatetimeIndex:
      - T_DOW_S, T_DOW_C
      - T_MIN_S, T_MIN_C (if intraday cadence detected)
      - T_MONTH_S, T_MONTH_C
      - T_DOY_S, T_DOY_C
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(idx)
            df.index = idx
        except Exception:
            return df

    # Detect intraday cadence
    deltas = idx.to_series().diff().dropna()
    intraday = False
    if not deltas.empty:
        step_delta = deltas.tail(min(1000, len(deltas))).median()
        intraday = step_delta < pd.Timedelta(hours=intraday_threshold_hours)

    def cyc(values, period):
        values = np.asarray(values, dtype=float)
        return np.sin(2 * np.pi * values / period).astype(np.float32), \
               np.cos(2 * np.pi * values / period).astype(np.float32)

    # Day of week
    dow = idx.weekday
    s, c = cyc(dow, 7.0)
    df["T_DOW_S"], df["T_DOW_C"] = s, c

    # Minute of day (only for intraday bars)
    if intraday:
        minute_of_day = idx.hour * 60 + idx.minute
        s, c = cyc(minute_of_day, 1440.0)
        df["T_MIN_S"], df["T_MIN_C"] = s, c

    # Month of year
    month = idx.month - 1
    s, c = cyc(month, 12.0)
    df["T_MONTH_S"], df["T_MONTH_C"] = s, c

    # Day of year
    doy = idx.dayofyear - 1
    s, c = cyc(doy, 366.0)
    df["T_DOY_S"], df["T_DOY_C"] = s, c

    return df


def time_series_train_test_split(X, y, test_ratio=0.2):
    n = len(X)
    split = int(n * (1 - test_ratio))
    return X[:split], X[split:], y[:split], y[split:]


def scale_train_test(X_train, X_test, y_train, y_test):
    """
    Fit scalers on TRAIN ONLY to avoid leakage. Scale X across features (flattened time),
    and y across target dimensions.
    """
    num_features = X_train.shape[2]
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_2d = X_train.reshape(-1, num_features)
    X_test_2d = X_test.reshape(-1, num_features)

    X_train_scaled = X_scaler.fit_transform(X_train_2d).reshape(X_train.shape)
    X_test_scaled = X_scaler.transform(X_test_2d).reshape(X_test.shape)

    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, X_scaler, y_scaler


def compute_metrics(y_true, y_pred, target_names):
    """
    y_true, y_pred in original price/volume scale.
    Prints MAE and RMSE per target and overall.
    """
    eps = 1e-12
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2 + eps, axis=0))
    print("Test metrics:")
    for i, name in enumerate(target_names):
        print(f"  {name}: MAE={mae[i]:.4f}, RMSE={rmse[i]:.4f}")
    print(f"Overall MAE={mae.mean():.4f}, RMSE={rmse.mean():.4f}")
    return mae, rmse


def infer_next_timestamp(df: pd.DataFrame):
    """
    Infers the next bar's timestamp from the DatetimeIndex.
    Uses the median of recent diffs as the step size.
    Returns (next_ts, step_delta) or (None, None) if not inferrable.
    """
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        try:
            idx = pd.to_datetime(idx)
        except Exception:
            return None, None

    if len(idx) < 2:
        return None, None

    deltas = idx.to_series().diff().dropna()
    if deltas.empty:
        return None, None

    step_delta = deltas.tail(min(1000, len(deltas))).median()
    next_ts = idx[-1] + step_delta
    return next_ts, step_delta

def model_shapes_compatible(model, seq_len, num_features, num_targets) -> bool:
    try:
        in_shape = model.input_shape  # (None, seq_len, num_features)
        out_shape = model.output_shape # (None, num_targets)
        return (in_shape[-2] == seq_len) and (in_shape[-1] == num_features) and (out_shape[-1] == num_targets)
    except Exception:
        return False

def scalers_compatible(X_scaler, Y_scaler, num_features, num_targets) -> bool:
    try:
        return (getattr(X_scaler, "n_features_in_", None) == num_features and
                getattr(Y_scaler, "n_features_in_", None) == num_targets)
    except Exception:
        return False
    
def ensure_parent_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)