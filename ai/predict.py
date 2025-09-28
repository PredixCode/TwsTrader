import os
import numpy as np
import joblib

from ai.price_predictor import PricePredictorLSTM
from ai.utils import fetch_price_df, add_mid_column, infer_next_timestamp



def predict_next_price(config):
    print("\n" + "="*20 + " NEXT OHLCV PREDICTION " + "="*20)
    df = fetch_price_df(config['TICKER'])
    if df.empty:
        print("No data available to predict.")
        return

    feature_cols = config['PRICE_FEATURES']  # includes Volume
    target_cols = config['PRICE_TARGETS']    # ["Open","High","Low","Close"]
    seq_len = config['PRICE_SEQUENCE_LENGTH']

    df = add_mid_column(df)
    if len(df) < seq_len + 1:
        print(f"Not enough data. Need at least {seq_len+1} rows, have {len(df)}.")
        return

    if not os.path.exists(config['PRICE_MODEL_PATH']):
        print(f"Model not found at {config['PRICE_MODEL_PATH']}. Train first.")
        return

    model = PricePredictorLSTM.load(config['PRICE_MODEL_PATH'])  # tf.keras.Model
    X_scaler = joblib.load(config['PRICE_SCALER_X_PATH'])
    Y_scaler = joblib.load(config['PRICE_SCALER_Y_PATH'])

    latest_window = df[feature_cols].values[-seq_len:]
    X_latest = np.expand_dims(latest_window, axis=0)

    num_features = X_latest.shape[2]
    X_latest_s = X_scaler.transform(X_latest.reshape(-1, num_features)).reshape(X_latest.shape)

    y_next_s = model.predict(X_latest_s, verbose=0)
    y_next = Y_scaler.inverse_transform(y_next_s)[0]  # shape (4,) for O/H/L/C

    # Infer next timestamp
    next_ts, step_delta = infer_next_timestamp(df)

    # Append Volume=0.0 for downstream consumers expecting OHLCV
    full_names = ["Open", "High", "Low", "Close", "Volume"]
    full_values = list(y_next) + [0.0]

    print(f"Next-step prediction for {config['TICKER']}:")
    if next_ts is None:
        print("  Datetime: (could not infer; insufficient or non-datetime index)")
    else:
        print(f"  Datetime: {next_ts} (step ≈ {step_delta})")
    for name, val in zip(full_names, full_values):
        print(f"  {name}: {val:.4f}")