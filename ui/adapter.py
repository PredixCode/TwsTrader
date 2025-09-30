import pandas as pd

class HubViewAdapter:
    """
    Adapter that presents a 'view-like' API to existing updaters (IBHistoryChartUpdater, YFinanceChartUpdater, TwsProvisionalCandleUpdater).
    It forwards into MarketDataHub with the right semantics:
      - apply_delta_df(...) => authoritative history delta
      - upsert_bar(...) => provisional=True for TWS provisional loop
      - remove_bar/delete_bar/drop_bar(...) => drop in hub
    """

    def __init__(self, hub):
        self.hub = hub

    # History updater calls this
    def apply_delta_df(self, delta_df):
        self.hub.apply_delta_df(delta_df)

    # Provisional updater calls this (no 'provisional' arg in caller)
    def upsert_bar(self, when, o, h, l, c, v, *, floor_to_minute: bool = True):
        ts = pd.Timestamp(when)
        if floor_to_minute:
            ts = ts.floor("T")
        # Mark as provisional at the hub
        self.hub.upsert_bar(ts, o, h, l, c, v, provisional=True)

    # Rollover drop path calls these various names
    def remove_bar(self, ts): self.hub.drop_bar(ts)
    def delete_bar(self, ts): self.hub.drop_bar(ts)
    def drop_bar(self, ts):   self.hub.drop_bar(ts)