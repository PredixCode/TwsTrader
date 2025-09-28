import os
import time
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from yfinance_wrapper.stock import FinanceStock



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


def fetch_price_df(ticker: str) -> pd.DataFrame:
    stock = FinanceStock(ticker)
    df = stock.get_historical_data()  # typically 7d @ 1m
    if df.empty:
        return df
    # Save last fetch as CSV
    stock.last_fetch_to_csv()

    cols_needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[cols_needed].dropna()
    return df


def add_mid_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Mid' not in df.columns:
        df['Mid'] = (df['High'] + df['Low']) / 2.0
    return df


def create_price_sequences(df: pd.DataFrame, sequence_length: int, feature_cols: list[str], target_cols: list[str], stride: int = 1):
    """
    Builds (X, y) where y is next-step targets for target_cols.
    y is shifted by -1 so each window predicts the next bar.
    """
    df = add_mid_column(df)  # harmless if not used in features/targets
    feat = df[feature_cols].values
    tgt = df[target_cols].shift(-1).dropna().values

    # Align features and targets (drop last feature row to match shifted targets)
    feat = feat[:len(tgt)]

    X, y = [], []
    for i in range(sequence_length, len(feat) + 1, stride):
        X.append(feat[i - sequence_length:i])
        y.append(tgt[i - 1])
    X = np.array(X)
    y = np.array(y)
    return X, y


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