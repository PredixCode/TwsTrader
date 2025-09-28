import os
import tensorflow as tf
from tensorflow.keras import layers, Model

class PricePredictorLSTM(Model):
    """
    Predicts next-step prices (multi-output regression) from a sliding window of OHLCV.
    """
    def __init__(self, seq_len: int, num_features: int, num_targets: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_targets = num_targets
        self.model = self._build()

    def _build(self) -> tf.keras.Model:
        inp = layers.Input(shape=(self.seq_len, self.num_features))
        x = layers.LSTM(128, return_sequences=True)(inp)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(64)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        out = layers.Dense(self.num_targets, activation='linear')(x)  # Linear for regression
        m = tf.keras.Model(inp, out, name="price_predictor_lstm")
        m.summary()
        return m

    def call(self, inputs):
        return self.model(inputs)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)

    @staticmethod
    def load(filepath: str):
        if not os.path.exists(filepath):
            print(f"Warning: Price model not found at {filepath}")
            return None
        return tf.keras.models.load_model(filepath)