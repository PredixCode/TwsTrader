import os
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from ai.price_predictor import PricePredictorLSTM
from ai.utils import set_random_seed, fetch_price_df, create_price_sequences, time_series_train_test_split, model_shapes_compatible, scalers_compatible, ensure_parent_dir



def run_price_training(config):
    print("\n" + "="*20 + " OHLCV PREDICTION TRAINING " + "="*20)
    set_random_seed(config['SEED'])

    df = fetch_price_df(config['TICKER'])
    if df.empty:
        print("No data available to train.")
        return

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

    # Determine whether to continue training
    can_try_continue = (not config.get("FRESH", False)
                        and os.path.exists(config['PRICE_MODEL_PATH'])
                        and os.path.exists(config['PRICE_SCALER_X_PATH'])
                        and os.path.exists(config['PRICE_SCALER_Y_PATH']))

    model = None
    continue_mode = False

    if can_try_continue:
        print("Found existing artifacts. Attempting to continue training...")
        try:
            loaded_model = PricePredictorLSTM.load(config['PRICE_MODEL_PATH'])  # tf.keras.Model
            X_scaler_loaded = joblib.load(config['PRICE_SCALER_X_PATH'])
            Y_scaler_loaded = joblib.load(config['PRICE_SCALER_Y_PATH'])
            if (loaded_model is not None
                and model_shapes_compatible(loaded_model, seq_len, X.shape[2], len(target_cols))
                and scalers_compatible(X_scaler_loaded, Y_scaler_loaded, X.shape[2], len(target_cols))):
                model = loaded_model
                X_scaler = X_scaler_loaded
                Y_scaler = Y_scaler_loaded
                continue_mode = True
                print("✅ Continuing training from saved model and scalers.")
            else:
                print("⚠️ Saved model/scalers are not compatible with current config. Training fresh.")
        except Exception as e:
            print(f"⚠️ Could not load previous model/scalers ({e}). Training fresh.")

    num_features = X_train.shape[2]
    if continue_mode:
        # Always keep scalers fixed when continuing
        X_train_s = X_scaler.transform(X_train.reshape(-1, num_features)).reshape(X_train.shape)
        X_test_s  = X_scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)
        y_train_s = Y_scaler.transform(y_train)
        y_test_s  = Y_scaler.transform(y_test)
    else:
        # Fresh: fit new scalers on current train split
        X_scaler = MinMaxScaler()
        Y_scaler = MinMaxScaler()
        X_train_s = X_scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(X_train.shape)
        X_test_s  = X_scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)
        y_train_s = Y_scaler.fit_transform(y_train)
        y_test_s  = Y_scaler.transform(y_test)

        # Build a fresh model wrapper that matches dims
        model_wrapper = PricePredictorLSTM(seq_len, num_features=X.shape[2], num_targets=len(target_cols))
        model = model_wrapper.model  # tf.keras.Model

    # Compile with appropriate LR
    lr = config['LR_CONTINUE'] if continue_mode else config['LR']
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
                  loss='huber', metrics=['mae'])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, min_delta=5e-7, restore_best_weights=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5, monitor='val_loss')
    ]

    model.fit(
        X_train_s, y_train_s,
        validation_data=(X_test_s, y_test_s),
        epochs=config['PRICE_EPOCHS'],
        batch_size=config['PRICE_BATCH_SIZE'],
        callbacks=callbacks,
        verbose=1
    )

    # Save artifacts
    ensure_parent_dir(config['PRICE_MODEL_PATH'])
    ensure_parent_dir(config['PRICE_SCALER_X_PATH'])    
    ensure_parent_dir(config['PRICE_SCALER_Y_PATH'])

    model.save(config['PRICE_MODEL_PATH'])
    joblib.dump(X_scaler, config['PRICE_SCALER_X_PATH'])
    joblib.dump(Y_scaler, config['PRICE_SCALER_Y_PATH'])
    
    print(f"Saved model to {config['PRICE_MODEL_PATH']}")
    print(f"Saved scalers to {config['PRICE_SCALER_X_PATH']} and {config['PRICE_SCALER_Y_PATH']}")
    print("Training complete.")