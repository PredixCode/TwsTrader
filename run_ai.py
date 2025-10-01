import argparse

from ai.utils import gpu_check
from ai.pipelines.train import run_price_training
from ai.pipelines.eval import evaluate_price_model
from ai.pipelines.predict import predict_next_price



def build_config(args):
    ticker = args.ticker or "RHM"
    ticker_file_name = ticker.replace("-","_").replace(".","_").replace(" ","")

    cfg = {
        "SEED": 42,
        "TICKER": ticker,
        "MAX_BARS": 200_000,
        "STRIDE": 1,
        "TEST_RATIO": 0.2,

        # Include Volume as both input and output
        "PRICE_FEATURES": ["Open", "High", "Low", "Close", "Volume"],
        "PRICE_TARGETS":  ["Open", "High", "Low", "Close", "Volume"],
        "BAR_SIZE": args.barSize,

        "PRICE_SEQUENCE_LENGTH": 90,
        "PRICE_EPOCHS": 50,
        "PRICE_BATCH_SIZE": 256,
        "LR": 3e-4,
        "LR_CONTINUE": 1e-4,

        "PRICE_MODEL_PATH": "ai/models/price_predictor.keras",
        "PRICE_SCALER_X_PATH": f"ai/models/scalers/{ticker_file_name}_X.joblib",
        "PRICE_SCALER_Y_PATH": f"ai/models/scalers/{ticker_file_name}_Y.joblib",
    }
    cfg["FRESH"] = bool(args.fresh)
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Next-step OHLCV prediction with a simple LSTM.")
    parser.add_argument('--ticker', type=str, default=None, help='Ticker symbol (e.g., RHM, BTC-EUR, AAPL, TSLA)')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'predict'],
                        help='Which stage to run.')
    parser.add_argument('--barSize', type=str, default='1m', help='The time for which one bar tracks OHCLV.')
    parser.add_argument('--evalMode', type=str, default='one_step',
                        choices=['one_step', 'autoreg'],
                        help='Evaluation mode. one_step uses actual data at each step; autoreg feeds predictions back.')
    parser.add_argument('--fresh', action='store_true', help='Ignore saved model/scalers and start fresh training')
    parser.add_argument('--gpu_check', type=bool, default=False, help='Check for available GPU at startup (debug)')
    args = parser.parse_args()

    config = build_config(args)

    if bool(args.gpu_check):
        gpu_check()

    if args.mode == 'train':
        run_price_training(config)
    elif args.mode == 'eval':
        evaluate_price_model(config, mode=args.evalMode)
    elif args.mode == 'predict':
        predict_next_price(config)


if __name__ == "__main__":
    main()