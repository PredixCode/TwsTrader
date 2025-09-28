import os
import time
import argparse
import numpy as np
import pandas as pd
import joblib

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from yfinance_wrapper.stock import FinanceStock
from ai.price_predictor import PricePredictorLSTM  # <-- use the shared model


# ============== Utilities ==============

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


def add_mid_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'Mid' not in df.columns:
        df['Mid'] = (df['High'] + df['Low']) / 2.0
    return df


def fetch_price_df(ticker: str) -> pd.DataFrame:
    stock = FinanceStock(ticker)
    df = stock.get_historical_data()  # typically 7d @ 1m
    if df.empty:
        return df
    # Save last fetch as CSV (as you added)
    stock.last_fetch_to_csv()

    # Basic OHLCV frame
    cols_needed = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[cols_needed].dropna()
    return df


def create_price_sequences(df: pd.DataFrame, sequence_length: int, feature_cols: list[str], target_cols: list[str], stride: int = 1):
    """
    Builds (X, y) where y is next-step targets for target_cols.
    y is shifted by -1 so each window predicts the next bar.
    """
    df = add_mid_column(df)  # harmless if not used in features/targets
    feat = df[feature_cols].values
    tgt = df[target_cols].shift(-1).dropna().values

    # Align features and targets (drop the last feature row to match shifted targets)
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

    # Prefer median of recent deltas to be robust to occasional gaps
    deltas = idx.to_series().diff().dropna()
    if deltas.empty:
        return None, None

    step_delta = deltas.tail(min(1000, len(deltas))).median()
    next_ts = idx[-1] + step_delta
    return next_ts, step_delta


# ============== Pipelines ==============

def run_price_training(config):
    print("\n" + "="*20 + " OHLCV PREDICTION TRAINING " + "="*20)
    set_random_seed(config['SEED'])

    df = fetch_price_df(config['TICKER'])
    if df.empty:
        print("No data available to train.")
        return

    # Optionally cap dataset size for quick iterations
    if config['MAX_BARS'] is not None and len(df) > config['MAX_BARS']:
        df = df.iloc[-config['MAX_BARS']:]

    feature_cols = config['PRICE_FEATURES']
    target_cols = config['PRICE_TARGETS']
    seq_len = config['PRICE_SEQUENCE_LENGTH']

    X, y = create_price_sequences(df, seq_len, feature_cols, target_cols, stride=config['STRIDE'])
    if len(X) < 1000:
        print(f"Warning: Very few sequences ({len(X)}). Consider a longer period or shorter seq_len.")
    print(f"Dataset: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = time_series_train_test_split(X, y, test_ratio=config['TEST_RATIO'])
    X_train_s, X_test_s, y_train_s, y_test_s, X_scaler, Y_scaler = scale_train_test(X_train, X_test, y_train, y_test)

    # Use the shared model class
    model_wrapper = PricePredictorLSTM(seq_len, num_features=X.shape[2], num_targets=len(target_cols))
    # Compile the underlying Keras model (keeps things explicit)
    model_wrapper.model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='huber', metrics=['mae'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5)
    ]

    model_wrapper.model.fit(
        X_train_s, y_train_s,
        validation_data=(X_test_s, y_test_s),
        epochs=config['PRICE_EPOCHS'],
        batch_size=config['PRICE_BATCH_SIZE'],
        callbacks=callbacks,
        verbose=1
    )

    # Save model + scalers
    os.makedirs(os.path.dirname(config['PRICE_MODEL_PATH']), exist_ok=True)
    model_wrapper.save(config['PRICE_MODEL_PATH'])
    joblib.dump(X_scaler, config['PRICE_SCALER_X_PATH'])
    joblib.dump(Y_scaler, config['PRICE_SCALER_Y_PATH'])
    print(f"Saved model to {config['PRICE_MODEL_PATH']}")
    print(f"Saved scalers to {config['PRICE_SCALER_X_PATH']} and {config['PRICE_SCALER_Y_PATH']}")
    print("Training complete.")


def evaluate_price_model(config):
    print("\n" + "="*20 + " OHLCV PREDICTION EVALUATION " + "="*20)
    df = fetch_price_df(config['TICKER'])
    if df.empty:
        print("No data available to evaluate.")
        return

    if config['MAX_BARS'] is not None and len(df) > config['MAX_BARS']:
        df = df.iloc[-config['MAX_BARS']:]

    feature_cols = config['PRICE_FEATURES']
    target_cols = config['PRICE_TARGETS']
    seq_len = config['PRICE_SEQUENCE_LENGTH']

    X, y = create_price_sequences(df, seq_len, feature_cols, target_cols, stride=config['STRIDE'])
    if len(X) == 0:
        print("No sequences generated. Check data and sequence length.")
        return

    X_train, X_test, y_train, y_test = time_series_train_test_split(X, y, test_ratio=config['TEST_RATIO'])

    # Load model and scalers
    if not os.path.exists(config['PRICE_MODEL_PATH']):
        print(f"Model not found at {config['PRICE_MODEL_PATH']}. Train first.")
        return
    model = PricePredictorLSTM.load(config['PRICE_MODEL_PATH'])  # returns a tf.keras.Model

    X_scaler = joblib.load(config['PRICE_SCALER_X_PATH'])
    Y_scaler = joblib.load(config['PRICE_SCALER_Y_PATH'])

    # Scale using loaded scalers
    num_features = X.shape[2]
    X_test_s = X_scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

    # Predict
    y_pred_s = model.predict(X_test_s, verbose=0)

    # Inverse to price/volume scale
    y_test_inv = Y_scaler.inverse_transform(y_test)
    y_pred_inv = Y_scaler.inverse_transform(y_pred_s)

    compute_metrics(y_test_inv, y_pred_inv, target_cols)

    # Optional: quick visualization on Close
    try:
        idx_close = target_cols.index('Close')
        plt.figure(figsize=(13, 5))
        plt.title(f"{config['TICKER']} - Close: Actual vs Predicted (Test slice)")
        plt.plot(y_test_inv[:300, idx_close], label='Actual Close', alpha=0.85)
        plt.plot(y_pred_inv[:300, idx_close], label='Predicted Close', alpha=0.85)
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Plotting skipped: {e}")


def predict_next_price(config):
    print("\n" + "="*20 + " NEXT OHLCV PREDICTION " + "="*20)
    df = fetch_price_df(config['TICKER'])
    if df.empty:
        print("No data available to predict.")
        return

    feature_cols = config['PRICE_FEATURES']
    target_cols = config['PRICE_TARGETS']
    seq_len = config['PRICE_SEQUENCE_LENGTH']

    df = add_mid_column(df)  # harmless if not used
    if len(df) < seq_len + 1:
        print(f"Not enough data. Need at least {seq_len+1} rows, have {len(df)}.")
        return

    # Load model & scalers
    if not os.path.exists(config['PRICE_MODEL_PATH']):
        print(f"Model not found at {config['PRICE_MODEL_PATH']}. Train first.")
        return
    model = PricePredictorLSTM.load(config['PRICE_MODEL_PATH'])  # returns a tf.keras.Model
    X_scaler = joblib.load(config['PRICE_SCALER_X_PATH'])
    Y_scaler = joblib.load(config['PRICE_SCALER_Y_PATH'])

    latest_window = df[feature_cols].values[-seq_len:]
    X_latest = np.expand_dims(latest_window, axis=0)

    num_features = X_latest.shape[2]
    X_latest_s = X_scaler.transform(X_latest.reshape(-1, num_features)).reshape(X_latest.shape)

    y_next_s = model.predict(X_latest_s, verbose=0)
    y_next = Y_scaler.inverse_transform(y_next_s)[0]

    # Infer and display the next timestamp
    next_ts, step_delta = infer_next_timestamp(df)
    print(f"Next-step prediction for {config['TICKER']}:")
    if next_ts is None:
        print("  Datetime: (could not infer; insufficient or non-datetime index)")
    else:
        print(f"  Datetime: {next_ts} (step ≈ {step_delta})")
    for i, name in enumerate(target_cols):
        print(f"  {name}: {y_next[i]:.4f}")


# ============== CLI ==============

def build_config(args):
    # Minimal, extendable config
    cfg = {
        "SEED": 42,
        "TICKER": args.ticker or "AAPL",
        "MAX_BARS": 200_000,       # Limit history for speed; set None for full
        "STRIDE": 1,               # Window step
        "TEST_RATIO": 0.2,

        # Use OHLCV for both features and targets
        "PRICE_FEATURES": ["Open", "High", "Low", "Close", "Volume"],
        "PRICE_TARGETS": ["Open", "High", "Low", "Close", "Volume"],

        "PRICE_SEQUENCE_LENGTH": 60,
        "PRICE_EPOCHS": 30,
        "PRICE_BATCH_SIZE": 128,

        "PRICE_MODEL_PATH": "ai/models/price_predictor.keras",
        "PRICE_SCALER_X_PATH": "ai/models/price_scaler_X.joblib",
        "PRICE_SCALER_Y_PATH": "ai/models/price_scaler_Y.joblib",
    }
    return cfg


def main():
    gpu_check()

    parser = argparse.ArgumentParser(description="Next-step OHLCV prediction with a simple LSTM.")
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'predict'],
                        help='Which stage to run.')
    parser.add_argument('--ticker', type=str, default=None, help='Ticker symbol (e.g., AAPL, TSLA)')
    args = parser.parse_args()

    config = build_config(args)

    if args.mode == 'train':
        run_price_training(config)
    elif args.mode == 'eval':
        evaluate_price_model(config)
    elif args.mode == 'predict':
        predict_next_price(config)


if __name__ == "__main__":
    main()