"""
UpsideOnly — Freqtrade Strategy v1.0
=====================================
Forgemaster's UpsideOnly logic ported to Freqtrade.

Rules:
- 2:1 R/R: ROI +0.5% / Stoploss -0.25%
- 15 min cooldown per pair after stoploss hit
- Flat filter: skip pairs with <0.15% price change (momentum check)
- Entry: EMA crossover + RSI confirmation
- High volatility pairs only (configured in pairlist)
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
from freqtrade.persistence import Trade


class UpsideOnly(IStrategy):
    """
    UpsideOnly v1.0
    Built for the Pantheon — paper only until validated.
    """

    # ── R/R ──────────────────────────────────────────────────────────────────
    minimal_roi = {
        "0": 0.005   # +0.5% take profit
    }

    stoploss = -0.0025  # -0.25% stoploss → 2:1 R/R

    # ── Timeframe ─────────────────────────────────────────────────────────────
    timeframe = "5m"

    # ── Trailing stop: off — fixed R/R is the law ─────────────────────────────
    trailing_stop = False

    # ── Cooldown after stoploss: 15 min (3 x 5m candles) ─────────────────────
    position_adjustment_enable = False

    # How many candles to wait after a loss before re-entering same pair
    # 15 min / 5 min = 3 candles
    ignore_roi_if_entry_signal = False

    # ── Startup candles needed for indicators ─────────────────────────────────
    startup_candle_count: int = 50

    # ── Hyperopt parameters (tune via freqtrade hyperopt) ─────────────────────
    ema_fast = IntParameter(8, 21, default=12, space="buy")
    ema_slow = IntParameter(21, 55, default=26, space="buy")
    rsi_entry = IntParameter(45, 65, default=55, space="buy")
    flat_threshold = DecimalParameter(0.001, 0.005, default=0.0015, space="buy")

    # ── Cooldown tracking ─────────────────────────────────────────────────────
    # Store last stoploss time per pair (in-memory, resets on restart)
    _stoploss_cooldown: dict = {}
    COOLDOWN_MINUTES = 15

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all indicators."""

        # EMA crossover
        dataframe["ema_fast"] = dataframe["close"].ewm(
            span=self.ema_fast.value, adjust=False
        ).mean()
        dataframe["ema_slow"] = dataframe["close"].ewm(
            span=self.ema_slow.value, adjust=False
        ).mean()

        # RSI
        delta = dataframe["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss
        dataframe["rsi"] = 100 - (100 / (1 + rs))

        # Price change % over last 24h candles (288 x 5m = 24h)
        dataframe["price_change_24h"] = dataframe["close"].pct_change(periods=288).abs()

        # Volume spike (2x average)
        dataframe["volume_ma"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["volume_spike"] = dataframe["volume"] > (dataframe["volume_ma"] * 1.5)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Entry signal: EMA cross up + RSI above threshold + not flat."""

        pair = metadata["pair"]
        now = datetime.utcnow()

        # Cooldown check — skip if we hit SL in last 15 min
        cooldown_until = self._stoploss_cooldown.get(pair)
        in_cooldown = cooldown_until is not None and now < cooldown_until

        conditions = [
            # EMA fast crosses above slow (momentum up)
            dataframe["ema_fast"] > dataframe["ema_slow"],
            dataframe["ema_fast"].shift(1) <= dataframe["ema_slow"].shift(1),

            # RSI confirms momentum
            dataframe["rsi"] > self.rsi_entry.value,
            dataframe["rsi"] < 80,  # not overbought

            # Not flat — minimum 0.15% move in last 24h
            dataframe["price_change_24h"] > self.flat_threshold.value,

            # Volume confirmation
            dataframe["volume_spike"],

            # Not in cooldown
            ~pd.Series([in_cooldown] * len(dataframe), index=dataframe.index),
        ]

        dataframe.loc[
            pd.concat([c if isinstance(c, pd.Series) else pd.Series(c, index=dataframe.index) 
                      for c in conditions], axis=1).all(axis=1),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit: EMA crosses back down or RSI exhausted. ROI/SL handle the R/R."""

        dataframe.loc[
            (
                (dataframe["ema_fast"] < dataframe["ema_slow"]) &
                (dataframe["rsi"] > 75)
            ),
            "exit_long",
        ] = 1

        return dataframe

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """Track stoploss hits for cooldown enforcement."""

        if exit_reason == "stop_loss":
            self._stoploss_cooldown[pair] = current_time + timedelta(
                minutes=self.COOLDOWN_MINUTES
            )

        return True
