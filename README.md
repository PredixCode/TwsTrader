### TWS Live Chart + Historical Backfill + AI Price Modeling + Incomming TradeBot (Realtime+Historical+Strategies+AI) 

A real-time charting and trading toolkit for Interactive Brokers (IB) that:
- Streams intra-minute prices and draws a provisional current bar.
- Backfills authoritative historical OHLCV on a schedule.
- Persists historical data to CSV.
- Trains and serves an LSTM model for next-step OHLCV predictions.
- Provides thin wrappers for placing IB orders.

---

### Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [IB Setup](#ib-setup)
- [Usage](#usage)
  - [Run from CLI](#run-from-cli)
  - [Run programmatically](#run-programmatically)
- [Architecture](#architecture)
  - [Components](#components)
  - [Data Flow](#data-flow)
  - [Display Offset](#display-offset)
- [Trading](#trading)
- [Historical Persistence](#historical-persistence)
- [AI Pipeline](#ai-pipeline)
  - [Configuration](#configuration)
  - [CLI Examples](#cli-examples)
- [Charting Tips](#charting-tips)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Disclaimer](#disclaimer)

---

### Overview

This project connects to IB TWS/IB Gateway via ib_insync, renders a live 1-minute chart with lightweight-charts, and continuously merges provisional intra-minute bars with authoritative historical OHLCV. It includes an AI pipeline to train, evaluate, and predict next-step OHLCV using LSTMs, plus simple order placement helpers for automated strategies.

Use cases:
- Visualize 1-minute bars with live intra-minute updates.
- Keep bars aligned and authoritative using periodic historical refresh.
- Save historical data for offline work.
- Train/evaluate/predict next-step OHLCV with a configurable LSTM.
- Place basic market/limit orders via Interactive Brokers.

---

### Features

- IB integration (ib_insync)
- Market data type discovery: Live → Frozen → Delayed → Delayed-Frozen
- Dual update loops:
  - Intra-minute updater (thread) builds a provisional bar from snapshots.
  - Historical updater (main thread) backfills authoritative bars with overlap.
- MarketDataHub merges provisional and historical data safely.
- Lightweight-charts UI (candles + optional volume; markers and labels).
- CSV persistence for last historical fetch.
- Basic trader (MKT/LMT, TIF, outsideRTH) with cancel helpers.
- AI pipelines: train, eval, predict next-step OHLCV.

---

### Requirements

- Python 3.10+ recommended
- Interactive Brokers TWS or IB Gateway with API enabled
- Market data permissions for your instruments

---

### Installation

#### Python dependencies

```bash
pip install ib_insync pandas numpy lightweight-charts joblib scikit-learn tensorflow
```

Notes:
- If you don’t need AI, you can skip tensorflow/scikit-learn/joblib.
- For GPU, install a TensorFlow variant compatible with your system.

---

### IB Setup

1. Start TWS or IB Gateway.
2. Enable API access: Configure → API → Settings → “Enable ActiveX and Socket Clients”.
3. Ensure host/port/clientId in your TwsConnection match your setup (e.g., 127.0.0.1:7497 for TWS paper).

---

### Usage

#### Run from CLI

```bash
python main.py
```

- You’ll be prompted for a symbol (default: RHM).
- The GUI will open the chart, start the intra-minute updater, and periodically refresh history.

#### Run programmatically

```python
from gui import GUI

gui = GUI(
    symbol="RHM",
    tz_offset_hours=+2.0,
    use_regular_trading_hours=False,
    intraminute_poll_secs=1.0,
    historical_poll_secs=60,
    price_pref=("mid", "last", "bid", "ask"),
    verbose=False,
)
gui.run()
```

Key parameters:
- tz_offset_hours: shifts display time (data stays UTC-naive internally).
- use_regular_trading_hours: history restricted to RTH if True.
- intraminute_poll_secs: frequency for live snapshot polling.
- historical_poll_secs: frequency for authoritative backfill.
- price_pref: price selection fallback chain for snapshots.

---

### Architecture

#### Components

##### GUI (gui.py)
- Orchestrates TWS connection, chart, MarketDataHub, and updaters.

##### TwsConnection (tws_wrapper/connection.py)
- Manages ib_insync IB instance and provides sleep to pump events.

##### TwsStock (tws_wrapper/stock.py)
- Contract qualification, market data subscription, snapshot(), historical fetch via cache, minTick, CSV persistence.
- Accurate max history: merges multi-res frames (5m → 2m → 1m, period='max') keeping the highest resolution.

##### TwsIntraMinuteUpdater (tws_wrapper/updater.py)
- Thread loop; builds the current minute’s provisional OHLCV from snapshots.
- On minute rollover, drops prior provisional bar (authoritative backfill replaces it later).

##### TwsHistoricUpdater (tws_wrapper/updater.py)
- Main-thread loop; periodically fetches history and pushes an overlap tail (e.g., last 5 minutes).
- Aligns to minute boundaries if requested; optional CSV persistence cadence.

##### MarketDataHub (core/market_data_hub.py)
- Canonical store; merges deltas, tracks provisional timestamps, applies display offset for views.
- Notifies views with incremental upserts or full redraw if needed.

##### TradeChart (ui/chart.py)
- Lightweight-charts wrapper for candles and optional volume histogram.
- Trade labels: markers, horizontal price lines, and short “stubs”.

##### TwsTrader (tws_wrapper/trader.py)
- Buy/sell (MKT/LMT), TIF, outsideRTH, simple account inference, cancel helpers, wait-for-fill.

##### Strategy stub (core/strategy/sfi_charlie.py)
- SfiCharlieStrategy and SfiCharlieConfig for future integration.

##### AI pipelines (ai/pipelines/*)
- run_price_training, evaluate_price_model, predict_next_price.

#### Data Flow

```text
Intra-minute (live):
TwsIntraMinuteUpdater (thread) → TwsStock.snapshot()
  → MarketDataHub.upsert_bar(provisional=True) → TradeChart

Historical (authoritative):
TwsHistoricUpdater (main) → TwsStock.get_historical_data()
  → MarketDataHub.apply_delta_df() → TradeChart

Minute rollover:
Intra-minute drops previous provisional minute; historical backfill overwrites with authoritative bars.
```

#### Display Offset

- MarketDataHub stores UTC-naive canonical timestamps.
- A display offset is applied only when notifying views (e.g., tz_offset_hours for local display without mutating canonical data).

---

### Trading

```python
from tws_wrapper.trader import TwsTrader
from tws_wrapper.connection import TwsConnection
from tws_wrapper.stock import TwsStock

conn = TwsConnection()
stock = TwsStock(connection=conn, symbol="RHM")
trader = TwsTrader(stock)

# Market buy 10 shares
trade = trader.buy(quantity=10, order_type='MKT', wait=True)
print(trader.trade_summary(trade))

# Limit sell 10 shares at 250.00
trade = trader.sell(quantity=10, order_type='LMT', limit_price=250.00, wait=True)
print(trader.trade_summary(trade))

# Cancel any remaining working orders for this contract
trader.cancel_all_for_contract()
```

**Caution:** Use paper trading first; confirm account, permissions, and risk controls.

---

### Historical Persistence

```python
# After a historical fetch
path = stock.last_fetch_to_csv()  # e.g., tws_wrapper/data/csv/RHM.csv
print("Saved to:", path)
```

---

### AI Pipeline

Train, evaluate, and predict next-step OHLCV (Open, High, Low, Close, Volume) using an LSTM.

#### Configuration

Defaults (via the CLI config builder):
- Features: ["Open", "High", "Low", "Close", "Volume"]
- Targets:  ["Open", "High", "Low", "Close", "Volume"]
- Sequence length: 90
- Epochs: 50
- Batch size: 256
- Learning rates: 3e-4 (train), 1e-4 (continue)
- Paths:
  - ai/models/price_predictor.keras
  - ai/models/scalers/<TICKER>_X.joblib
  - ai/models/scalers/<TICKER>_Y.joblib

#### CLI Examples

```bash
# Train from scratch
python ai/cli.py --ticker RHM --mode train --barSize 1m --fresh

# Evaluate (one-step)
python ai/cli.py --ticker RHM --mode eval --barSize 1m --evalMode one_step

# Predict next bar
python ai/cli.py --ticker RHM --mode predict --barSize 1m
```

Notes:
- Replace ai/cli.py with your actual entry point if named differently.
- Add --gpu_check True to print GPU availability (debug).

---

### Charting Tips

#### Add labeled trade markers and lines

```python
chart.add_trade_label(
    when=None,             # latest bar
    side="buy",
    price=247.35,
    text="BUY @ 247.35",
    use_marker=True,
    show_price_label=True,
    show_price_stub=True,
    stub_bars=2,
    line_style='dashed',
    line_width=1,
)
```

#### Incremental updates

- TradeChart tries candles.update/volume.update for efficiency.
- Falls back to a full redraw when needed to keep candles/volume/markers consistent.

---

### Project Structure

```text
main.py
gui.py
ui/
  chart.py
core/
  market_data_hub.py
  strategy/
    sfi_charlie.py
tws_wrapper/
  connection.py
  stock.py
  cache.py
  trader.py
ai/
  pipelines/
    train.py
    eval.py
    predict.py
  models/
  utils.py
```

(Names inferred from imports; adjust to match your repository.)

---

### Troubleshooting

#### No market data
- Verify symbol, exchange, currency, and your IB market data permissions.
- Check market data type discovery logs (you may only have delayed data).

#### Connection issues
- Ensure TWS/IB Gateway is running, API enabled, and host/port/clientId match.

#### Empty chart
- Confirm historical fetch returns rows and contract qualification succeeded.

#### UI not updating
- Intra-minute thread running? Historical updater ticking?
- Ensure ib_insync loop is pumped (during_wait uses ib.sleep()).

#### TensorFlow errors
- Install a TF build compatible with your OS/Python.
- Use --gpu_check to verify GPU detection.

---

### Roadmap

- Live chart:
  - Connect trading bot to live chart.
  - Overlay AI predictions and signals.
- AI:
  - Train generalized or Rheinmetall-specific model.
  - Feed AI predictions into strategies.

---

### Disclaimer

This project is for educational and research purposes. Markets involve risk. Use paper trading first, validate thoroughly, and comply with your broker and local regulations.
