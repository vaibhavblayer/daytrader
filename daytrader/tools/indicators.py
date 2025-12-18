"""Indicator calculation tools for AI agents.

These tools allow agents to calculate technical indicators and
find support/resistance levels from price data.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from daytrader.db.store import DataStore
from daytrader.indicators.technical import (
    calculate_rsi,
    calculate_macd,
    calculate_ema,
    calculate_sma,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_vwap,
)


# Default database path
DEFAULT_DB_PATH = Path.home() / ".config" / "daytrader" / "daytrader.db"


def _get_data_store(db_path: Optional[Path] = None) -> DataStore:
    """Get a DataStore instance."""
    return DataStore(db_path or DEFAULT_DB_PATH)


def _is_valid(value: float) -> bool:
    """Check if a value is valid (not NaN)."""
    return value == value  # NaN != NaN


def calculate_indicators(
    symbol: str,
    indicators: Optional[list[str]] = None,
    timeframe: str = "1day",
    days: int = 100,
    db_path: Optional[Path] = None,
) -> dict:
    """Calculate technical indicators for a symbol.
    
    This tool calculates various technical indicators using price data
    from the SQLite database.
    
    Args:
        symbol: Trading symbol (e.g., "RELIANCE", "TCS").
        indicators: List of indicators to calculate. Options:
                   "rsi", "macd", "ema", "sma", "bb" (Bollinger Bands),
                   "atr", "vwap". If None, calculates all.
        timeframe: Candle timeframe (default "1day").
        days: Number of days of data to use (default 100).
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - symbol: The queried symbol
        - timeframe: The timeframe used
        - data_points: Number of candles used
        - indicators: Dictionary of calculated indicator values
        - signals: Trading signals based on indicators
        - error: Error message if calculation failed (None if successful)
    """
    try:
        store = _get_data_store(db_path)
        
        # Get candle data
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        candles = store.get_candles(
            symbol=symbol.upper(),
            timeframe=timeframe,
            from_date=start_date,
            to_date=end_date,
        )
        
        if not candles:
            return {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "data_points": 0,
                "indicators": {},
                "signals": {},
                "error": f"No data found for {symbol.upper()} in timeframe {timeframe}",
            }
        
        # Extract price arrays
        closes = [c.close for c in candles]
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        volumes = [float(c.volume) for c in candles]
        
        # Default to all indicators if none specified
        if indicators is None:
            indicators = ["rsi", "macd", "ema", "sma", "bb", "atr", "vwap"]
        
        indicators = [i.lower() for i in indicators]
        
        result_indicators = {}
        signals = {}
        
        # Calculate RSI
        if "rsi" in indicators:
            rsi_values = calculate_rsi(closes, period=14)
            latest_rsi = next(
                (v for v in reversed(rsi_values) if _is_valid(v)), None
            )
            result_indicators["rsi"] = {
                "value": latest_rsi,
                "period": 14,
            }
            if latest_rsi is not None:
                if latest_rsi < 30:
                    signals["rsi"] = "oversold"
                elif latest_rsi > 70:
                    signals["rsi"] = "overbought"
                else:
                    signals["rsi"] = "neutral"
        
        # Calculate MACD
        if "macd" in indicators:
            macd_line, signal_line, histogram = calculate_macd(closes)
            latest_macd = next(
                (v for v in reversed(macd_line) if _is_valid(v)), None
            )
            latest_signal = next(
                (v for v in reversed(signal_line) if _is_valid(v)), None
            )
            latest_hist = next(
                (v for v in reversed(histogram) if _is_valid(v)), None
            )
            result_indicators["macd"] = {
                "macd_line": latest_macd,
                "signal_line": latest_signal,
                "histogram": latest_hist,
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9,
            }
            if latest_macd is not None and latest_signal is not None:
                if latest_macd > latest_signal:
                    signals["macd"] = "bullish"
                else:
                    signals["macd"] = "bearish"
        
        # Calculate EMA
        if "ema" in indicators:
            ema_20 = calculate_ema(closes, period=20)
            ema_50 = calculate_ema(closes, period=50)
            latest_ema_20 = next(
                (v for v in reversed(ema_20) if _is_valid(v)), None
            )
            latest_ema_50 = next(
                (v for v in reversed(ema_50) if _is_valid(v)), None
            )
            result_indicators["ema"] = {
                "ema_20": latest_ema_20,
                "ema_50": latest_ema_50,
            }
            if latest_ema_20 is not None and latest_ema_50 is not None:
                if latest_ema_20 > latest_ema_50:
                    signals["ema"] = "bullish"
                else:
                    signals["ema"] = "bearish"
        
        # Calculate SMA
        if "sma" in indicators:
            sma_20 = calculate_sma(closes, period=20)
            sma_50 = calculate_sma(closes, period=50)
            latest_sma_20 = next(
                (v for v in reversed(sma_20) if _is_valid(v)), None
            )
            latest_sma_50 = next(
                (v for v in reversed(sma_50) if _is_valid(v)), None
            )
            result_indicators["sma"] = {
                "sma_20": latest_sma_20,
                "sma_50": latest_sma_50,
            }
        
        # Calculate Bollinger Bands
        if "bb" in indicators:
            upper, middle, lower = calculate_bollinger_bands(closes)
            latest_upper = next(
                (v for v in reversed(upper) if _is_valid(v)), None
            )
            latest_middle = next(
                (v for v in reversed(middle) if _is_valid(v)), None
            )
            latest_lower = next(
                (v for v in reversed(lower) if _is_valid(v)), None
            )
            result_indicators["bollinger_bands"] = {
                "upper": latest_upper,
                "middle": latest_middle,
                "lower": latest_lower,
                "period": 20,
                "std_dev": 2.0,
            }
            if latest_upper and latest_lower and closes:
                current_price = closes[-1]
                if current_price > latest_upper:
                    signals["bb"] = "overbought"
                elif current_price < latest_lower:
                    signals["bb"] = "oversold"
                else:
                    signals["bb"] = "neutral"
        
        # Calculate ATR
        if "atr" in indicators:
            atr_values = calculate_atr(highs, lows, closes, period=14)
            latest_atr = next(
                (v for v in reversed(atr_values) if _is_valid(v)), None
            )
            result_indicators["atr"] = {
                "value": latest_atr,
                "period": 14,
            }
        
        # Calculate VWAP
        if "vwap" in indicators:
            vwap_values = calculate_vwap(highs, lows, closes, volumes)
            latest_vwap = next(
                (v for v in reversed(vwap_values) if _is_valid(v)), None
            )
            result_indicators["vwap"] = {
                "value": latest_vwap,
            }
            if latest_vwap and closes:
                current_price = closes[-1]
                if current_price > latest_vwap:
                    signals["vwap"] = "above_vwap"
                else:
                    signals["vwap"] = "below_vwap"
        
        # Add current price for reference
        result_indicators["current_price"] = closes[-1] if closes else None
        
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "data_points": len(candles),
            "indicators": result_indicators,
            "signals": signals,
            "error": None,
        }
        
    except Exception as e:
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "data_points": 0,
            "indicators": {},
            "signals": {},
            "error": str(e),
        }


def find_support_resistance(
    symbol: str,
    timeframe: str = "1day",
    days: int = 60,
    num_levels: int = 3,
    db_path: Optional[Path] = None,
) -> dict:
    """Find support and resistance levels for a symbol.
    
    This tool identifies key support and resistance levels based on
    historical price data using pivot points and price clustering.
    
    Args:
        symbol: Trading symbol (e.g., "RELIANCE", "TCS").
        timeframe: Candle timeframe (default "1day").
        days: Number of days of data to analyze (default 60).
        num_levels: Number of support/resistance levels to return (default 3).
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - symbol: The queried symbol
        - current_price: Current price
        - support_levels: List of support levels (sorted descending)
        - resistance_levels: List of resistance levels (sorted ascending)
        - pivot_point: Classic pivot point
        - error: Error message if calculation failed (None if successful)
    """
    try:
        store = _get_data_store(db_path)
        
        # Get candle data
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        candles = store.get_candles(
            symbol=symbol.upper(),
            timeframe=timeframe,
            from_date=start_date,
            to_date=end_date,
        )
        
        if not candles:
            return {
                "symbol": symbol.upper(),
                "current_price": None,
                "support_levels": [],
                "resistance_levels": [],
                "pivot_point": None,
                "error": f"No data found for {symbol.upper()} in timeframe {timeframe}",
            }
        
        # Extract price data
        highs = [c.high for c in candles]
        lows = [c.low for c in candles]
        closes = [c.close for c in candles]
        
        current_price = closes[-1]
        
        # Calculate classic pivot point using most recent complete candle
        if len(candles) >= 2:
            prev_candle = candles[-2]
            pivot = (prev_candle.high + prev_candle.low + prev_candle.close) / 3
            
            # Classic pivot levels
            r1 = 2 * pivot - prev_candle.low
            r2 = pivot + (prev_candle.high - prev_candle.low)
            r3 = prev_candle.high + 2 * (pivot - prev_candle.low)
            
            s1 = 2 * pivot - prev_candle.high
            s2 = pivot - (prev_candle.high - prev_candle.low)
            s3 = prev_candle.low - 2 * (prev_candle.high - pivot)
        else:
            pivot = current_price
            r1 = r2 = r3 = current_price
            s1 = s2 = s3 = current_price
        
        # Find swing highs and lows for additional levels
        swing_highs = []
        swing_lows = []
        
        lookback = 5  # Number of candles to look back/forward for swing detection
        
        for i in range(lookback, len(candles) - lookback):
            # Check for swing high
            is_swing_high = all(
                highs[i] >= highs[j]
                for j in range(i - lookback, i + lookback + 1)
                if j != i
            )
            if is_swing_high:
                swing_highs.append(highs[i])
            
            # Check for swing low
            is_swing_low = all(
                lows[i] <= lows[j]
                for j in range(i - lookback, i + lookback + 1)
                if j != i
            )
            if is_swing_low:
                swing_lows.append(lows[i])
        
        # Combine pivot levels with swing levels
        all_resistance = [r1, r2, r3] + [h for h in swing_highs if h > current_price]
        all_support = [s1, s2, s3] + [l for l in swing_lows if l < current_price]
        
        # Remove duplicates and sort
        resistance_levels = sorted(set(
            round(r, 2) for r in all_resistance if r > current_price
        ))[:num_levels]
        
        support_levels = sorted(set(
            round(s, 2) for s in all_support if s < current_price
        ), reverse=True)[:num_levels]
        
        return {
            "symbol": symbol.upper(),
            "current_price": round(current_price, 2),
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "pivot_point": round(pivot, 2),
            "r1": round(r1, 2),
            "r2": round(r2, 2),
            "s1": round(s1, 2),
            "s2": round(s2, 2),
            "error": None,
        }
        
    except Exception as e:
        return {
            "symbol": symbol.upper(),
            "current_price": None,
            "support_levels": [],
            "resistance_levels": [],
            "pivot_point": None,
            "error": str(e),
        }
