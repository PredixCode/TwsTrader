# ui/adapter.py
from core.normalize import normalize_ts

class HubViewAdapter:
    """
    Adapter that presents a 'view-like' API to existing updaters (IBHistoryChartUpdater, YFinanceChartUpdater, TwsProvisionalCandleUpdater).
    Forwards into MarketDataHub with correct semantics and unified normalization.
    """

    def __init__(self, hub):
        self.hub = hub

    # Authoritative history delta (already DataFrame-level; hub will normalize)
    def apply_delta_df(self, delta_df):
        self.hub.apply_delta_df(delta_df)

    # Provisional: normalize ts consistently and mark as provisional
    def upsert_bar(self, when, o, h, l, c, v, *, floor_to_minute: bool = True):
        ts = normalize_ts(when, floor_to_minute=floor_to_minute)
        self.hub.upsert_bar(ts, o, h, l, c, v, provisional=True)

    # Rollover drop aliases — always floor to minute for 1m bars
    def remove_bar(self, ts):
        self.hub.drop_bar(normalize_ts(ts, floor_to_minute=True))

    def delete_bar(self, ts):
        self.hub.drop_bar(normalize_ts(ts, floor_to_minute=True))

    def drop_bar(self, ts):
        self.hub.drop_bar(normalize_ts(ts, floor_to_minute=True))