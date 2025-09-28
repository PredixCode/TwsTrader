# ai/evaluate.py
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

from ai.price_predictor import PricePredictorLSTM
from ai.utils import (
    fetch_price_df,
    create_price_sequences,
    time_series_train_test_split,
    model_shapes_compatible,
    scalers_compatible,
)

def evaluate_price_model(config, mode: str = 'one_step'):
    """
    Evaluate the LSTM price model.

    Modes:
      - one_step: teacher-forced next-step predictions (uses true windows).
      - autoreg:  open-loop rollout with predictions fed back into inputs.

    Config keys expected:
      - TICKER, MAX_BARS
      - PRICE_FEATURES (list[str]), PRICE_TARGETS (list[str])
      - PRICE_SEQUENCE_LENGTH (int), STRIDE (int), TEST_RATIO (float)
      - PRICE_MODEL_PATH, PRICE_SCALER_X_PATH, PRICE_SCALER_Y_PATH
      - Optional:
          AUTOREG_SEED: 'test' (default) or 'latest'
          AUTOREG_STEPS: int or None (default: len(y_test) in 'test' mode; or pick a small number for 'latest')
    """
    print("\n" + "="*20 + f" OHLC PREDICTION EVALUATION [{mode}] " + "="*20)

    # 1) Load recent data
    df = fetch_price_df(config['TICKER'])
    if df.empty:
        print("No data available to evaluate.")
        return

    if config.get('MAX_BARS') is not None and len(df) > config['MAX_BARS']:
        df = df.iloc[-config['MAX_BARS']:]

    feature_cols = list(config['PRICE_FEATURES'])
    target_cols  = list(config['PRICE_TARGETS'])
    seq_len      = int(config['PRICE_SEQUENCE_LENGTH'])
    stride       = int(config['STRIDE'])
    test_ratio   = float(config['TEST_RATIO'])

    # 2) Build raw sequences and split
    X, y = create_price_sequences(df, seq_len, feature_cols, target_cols, stride=stride)
    if len(X) == 0:
        print("No sequences generated. Check data and sequence length.")
        return

    X_train, X_test, y_train, y_test = time_series_train_test_split(X, y, test_ratio=test_ratio)

    # 3) Load model and scalers
    model_path   = config['PRICE_MODEL_PATH']
    x_scaler_path = config['PRICE_SCALER_X_PATH']
    y_scaler_path = config['PRICE_SCALER_Y_PATH']

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Train first.")
        return

    model = PricePredictorLSTM.load(model_path)  # tf.keras.Model
    X_scaler = joblib.load(x_scaler_path)
    Y_scaler = joblib.load(y_scaler_path)

    num_features = len(feature_cols)
    num_targets  = len(target_cols)

    # Compatibility checks
    if model is None:
        print("Failed to load model.")
        return
    if not model_shapes_compatible(model, seq_len, num_features, num_targets):
        print("Model shapes not compatible with config. Check seq_len, features, and targets.")
        return
    if not scalers_compatible(X_scaler, Y_scaler, num_features, num_targets):
        print("Scaler shapes not compatible with config. Re-train or re-save scalers.")
        return

    # 4) Evaluate
    if mode == 'autoreg':
        preds, y_true = _evaluate_autoreg(
            model, X_scaler, Y_scaler,
            df=df,
            X_test=X_test, y_test=y_test,
            feature_cols=feature_cols, target_cols=target_cols,
            seq_len=seq_len,
            seed_mode=str(config.get('AUTOREG_SEED', 'test')),
            steps=config.get('AUTOREG_STEPS', None)
        )
    else:
        preds, y_true = _evaluate_one_step(model, X_scaler, Y_scaler, X_test, y_test)

    # 5) Metrics
    _print_metrics(y_true, preds, target_cols)

    # 6) Plot Close
    _plot_close(config['TICKER'], target_cols, y_true, preds, mode)


def _evaluate_one_step(model, X_scaler, Y_scaler, X_test, y_test):
    """Teacher-forced next-step evaluation."""
    num_features = X_test.shape[2]
    X_test_s = X_scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)
    y_pred_s = model.predict(X_test_s, verbose=0)
    preds    = Y_scaler.inverse_transform(y_pred_s)
    y_true   = y_test  # already raw
    return preds.astype(np.float32), y_true.astype(np.float32)


def _evaluate_autoreg(
    model, X_scaler, Y_scaler,
    df, X_test, y_test,
    feature_cols, target_cols,
    seq_len,
    seed_mode='test',
    steps=None
):
    """
    Open-loop rollout:
      - Pick a seed window.
      - Predict one step, de-scale to raw, build next feature row by inserting predicted targets
        and carrying forward other features (including Volume).
      - Slide window and repeat.

    Returns preds (raw) and y_true aligned to the same timeline when seed_mode='test'.
    If seed_mode='latest', y_true will be truncated/empty unless you supply a small backtest horizon.
    """
    num_features = len(feature_cols)
    num_targets  = len(target_cols)

    # Seed
    if seed_mode == 'latest':
        # Use latest real data window for forward rollout
        window_raw = df[feature_cols].values.astype(np.float32)[-seq_len:].copy()
        if steps is None:
            steps = min(len(y_test), 200)  # heuristic default for forward rollout
        y_true = np.zeros((steps, num_targets), dtype=np.float32)  # no ground truth for the future
    else:
        # Use first test window: backtest with aligned y_true
        window_raw = X_test[0].astype(np.float32).copy()
        if steps is None:
            steps = len(y_test)
        y_true = y_test[:steps].astype(np.float32)

    preds = np.zeros((steps, num_targets), dtype=np.float32)

    for t in range(steps):
        # Scale window and predict next target vector
        w_scaled = X_scaler.transform(window_raw.reshape(-1, num_features)).reshape(1, seq_len, num_features)
        y_next_s = model.predict(w_scaled, verbose=0)            # (1, num_targets), scaled
        y_next   = Y_scaler.inverse_transform(y_next_s)[0]       # (num_targets,), raw
        preds[t] = y_next

        # Build next feature row in RAW space
        next_feat_row = _build_next_feature_row(
            y_next_vec=y_next,
            last_row_vec=window_raw[-1],
            feature_cols=feature_cols,
            target_cols=target_cols,
            carry_forward_non_targets=True
        )
        # Slide window
        window_raw = np.vstack([window_raw[1:], next_feat_row])

    return preds.astype(np.float32), y_true


def _build_next_feature_row(
    y_next_vec,
    last_row_vec,
    feature_cols,
    target_cols,
    carry_forward_non_targets=True
):
    """
    Combine predicted targets with last observed non-target features to form the next feature row (raw scale).
    - Always carry-forward non-tar features (including Volume) unless a different strategy is needed.
    """
    tgt_map = dict(zip(target_cols, y_next_vec))
    out = []
    for j, col in enumerate(feature_cols):
        if col in tgt_map:
            out.append(float(tgt_map[col]))
        elif carry_forward_non_targets:
            out.append(float(last_row_vec[j]))      # carry forward (incl. Volume)
        else:
            out.append(float(last_row_vec[j]))      # default fallback
    return np.array(out, dtype=np.float32)


def _print_metrics(y_true, preds, target_cols):
    mae  = np.mean(np.abs(y_true - preds), axis=0)
    rmse = np.sqrt(np.mean((y_true - preds) ** 2, axis=0))
    print("Evaluation metrics:")
    for i, name in enumerate(target_cols):
        print(f"  {name}: MAE={mae[i]:.4f}, RMSE={rmse[i]:.4f}")
    print(f"Overall MAE={mae.mean():.4f}, RMSE={rmse.mean():.4f}")


def _plot_close(ticker, target_cols, y_true, preds, mode):
    try:
        if "Close" in target_cols:
            idx_close = target_cols.index('Close')
            nplot = min(300, len(y_true))
            plt.figure(figsize=(13, 5))
            title_mode = "Autoregressive" if mode == 'autoreg' else "One-step (teacher-forced)"
            plt.title(f"{ticker} - Close: {title_mode} Actual vs Predicted (first {nplot} steps)")
            if len(y_true) > 0:
                plt.plot(y_true[:nplot, idx_close], label='Actual Close', alpha=0.85)
            plt.plot(preds[:nplot, idx_close], label='Predicted Close', alpha=0.85)
            plt.legend(); plt.grid(True); plt.show()
    except Exception as e:
        print(f"Plotting skipped: {e}")