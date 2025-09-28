import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from ai.price_predictor import PricePredictorLSTM
from ai.utils import fetch_price_df, create_price_sequences, time_series_train_test_split



def evaluate_price_model(config, mode: str = 'one_step'):
    """
    Evaluate the model.
    - one_step: teacher-forced next-step predictions (uses actual sequence windows from data).
    - autoreg:  open-loop rollout (predictions are fed back). Kept as a toggle.
    Notes:
      * Only targets in config['PRICE_TARGETS'] are predicted (e.g., ["Open","High","Low","Close"]).
      * PRICE_FEATURES can be without Volume or include Volume. We never predict Volume.
    """
    print("\n" + "="*20 + f" OHLC PREDICTION EVALUATION [{mode}] " + "="*20)

    df = fetch_price_df(config['TICKER'])
    if df.empty:
        print("No data available to evaluate.")
        return

    if config['MAX_BARS'] is not None and len(df) > config['MAX_BARS']:
        df = df.iloc[-config['MAX_BARS']:]

    feature_cols = config['PRICE_FEATURES']     # e.g. ["Open","High","Low","Close"]
    target_cols  = config['PRICE_TARGETS']      # e.g. ["Open","High","Low","Close"]
    seq_len      = config['PRICE_SEQUENCE_LENGTH']
    num_features = len(feature_cols)
    num_targets  = len(target_cols)

    # Build raw (unscaled) sequences and split
    X, y = create_price_sequences(df, seq_len, feature_cols, target_cols, stride=config['STRIDE'])
    if len(X) == 0:
        print("No sequences generated. Check data and sequence length.")
        return
    _, X_test, _, y_test = time_series_train_test_split(X, y, test_ratio=config['TEST_RATIO'])

    # Load model + per-ticker scalers
    if not os.path.exists(config['PRICE_MODEL_PATH']):
        print(f"Model not found at {config['PRICE_MODEL_PATH']}. Train first.")
        return
    model    = PricePredictorLSTM.load(config['PRICE_MODEL_PATH'])  # tf.keras.Model
    X_scaler = joblib.load(config['PRICE_SCALER_X_PATH'])
    Y_scaler = joblib.load(config['PRICE_SCALER_Y_PATH'])

    # Helper used only for autoreg
    def build_next_feat_row(y_next_vec, last_row_vec):
        tgt_map = dict(zip(target_cols, y_next_vec))
        out = []
        for col in feature_cols:
            if col in tgt_map:
                out.append(float(tgt_map[col]))
            elif col.lower() == "volume":
                out.append(0.0)  # we do not predict volume
            else:
                j = feature_cols.index(col)
                out.append(float(last_row_vec[j]))  # carry forward any other non-target feature
        return np.array(out, dtype=np.float32)

    if mode == 'autoreg':
        # Open-loop rollout: predictions fed back
        window_raw = X_test[0].astype(np.float32).copy()     # shape: (seq_len, num_features)
        steps = len(y_test)
        preds = np.zeros((steps, num_targets), dtype=np.float32)

        for t in range(steps):
            w_scaled = X_scaler.transform(window_raw.reshape(-1, num_features)).reshape(1, seq_len, num_features)
            y_next_s = model.predict(w_scaled, verbose=0)
            y_next   = Y_scaler.inverse_transform(y_next_s)[0]
            preds[t] = y_next
            next_feat_row = build_next_feat_row(y_next, window_raw[-1])
            window_raw = np.vstack([window_raw[1:], next_feat_row])

        y_true = y_test[:steps]

    else:
        # Teacher-forced one-step: evaluate on actual windows
        num_features = X_test.shape[2]
        X_test_s = X_scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)
        y_pred_s = model.predict(X_test_s, verbose=0)
        preds    = Y_scaler.inverse_transform(y_pred_s)      # back to raw scale
        y_true   = y_test                                    # already raw

    # Metrics
    mae  = np.mean(np.abs(y_true - preds), axis=0)
    rmse = np.sqrt(np.mean((y_true - preds) ** 2, axis=0))
    print("Evaluation metrics:")
    for i, name in enumerate(target_cols):
        print(f"  {name}: MAE={mae[i]:.4f}, RMSE={rmse[i]:.4f}")
    print(f"Overall MAE={mae.mean():.4f}, RMSE={rmse.mean():.4f}")

    # Plot Close
    try:
        if "Close" in target_cols:
            idx_close = target_cols.index('Close')
            nplot = min(300, len(y_true))
            plt.figure(figsize=(13, 5))
            title_mode = "Autoregressive" if mode == 'autoreg' else "One-step (teacher-forced)"
            plt.title(f"{config['TICKER']} - Close: {title_mode} Actual vs Predicted (first {nplot} steps)")
            plt.plot(y_true[:nplot, idx_close], label='Actual Close', alpha=0.85)
            plt.plot(preds[:nplot, idx_close], label='Predicted Close', alpha=0.85)
            plt.legend(); plt.grid(True); plt.show()
    except Exception as e:
        print(f"Plotting skipped: {e}")