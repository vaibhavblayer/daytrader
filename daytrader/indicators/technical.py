"""Technical indicator calculations for trading analysis.

This module provides functions to calculate common technical indicators
used in stock market analysis. Calculations are validated against pandas-ta
as a reference implementation.
"""

from typing import Optional


def calculate_sma(prices: list[float], period: int) -> list[float]:
    """Calculate Simple Moving Average.
    
    Args:
        prices: List of price values (typically close prices)
        period: Number of periods for the moving average
        
    Returns:
        List of SMA values. First (period-1) values will be NaN.
    """
    if len(prices) < period or period < 1:
        return [float('nan')] * len(prices)
    
    result = [float('nan')] * (period - 1)
    
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1:i + 1]
        result.append(sum(window) / period)
    
    return result


def calculate_ema(prices: list[float], period: int) -> list[float]:
    """Calculate Exponential Moving Average.
    
    Args:
        prices: List of price values
        period: Number of periods for the EMA
        
    Returns:
        List of EMA values. First (period-1) values will be NaN.
    """
    if len(prices) < period or period < 1:
        return [float('nan')] * len(prices)
    
    result = [float('nan')] * (period - 1)
    multiplier = 2 / (period + 1)
    
    # First EMA is SMA
    first_sma = sum(prices[:period]) / period
    result.append(first_sma)
    
    # Calculate subsequent EMAs
    for i in range(period, len(prices)):
        ema = (prices[i] - result[-1]) * multiplier + result[-1]
        result.append(ema)
    
    return result


def calculate_rsi(prices: list[float], period: int = 14) -> list[float]:
    """Calculate Relative Strength Index.
    
    Uses exponential weighted mean (EWM) with alpha=1/period to match
    pandas-ta reference implementation.
    
    Args:
        prices: List of price values (typically close prices)
        period: RSI period (default 14)
        
    Returns:
        List of RSI values (0-100). First `period` values will be NaN.
    """
    if len(prices) < period + 1 or period < 1:
        return [float('nan')] * len(prices)
    
    # Calculate price changes
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    
    gains = [max(0, c) for c in changes]
    losses = [abs(min(0, c)) for c in changes]
    
    # Use EWM with alpha = 1/period (same as pandas-ta)
    alpha = 1.0 / period
    
    # Initialize with first value
    avg_gain = gains[0]
    avg_loss = losses[0]
    
    # Build up EWM averages
    avg_gains = [avg_gain]
    avg_losses = [avg_loss]
    
    for i in range(1, len(changes)):
        avg_gain = alpha * gains[i] + (1 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1 - alpha) * avg_loss
        avg_gains.append(avg_gain)
        avg_losses.append(avg_loss)
    
    # Calculate RSI - first `period` values are NaN (need min_periods)
    result = [float('nan')] * period
    
    for i in range(period - 1, len(changes)):
        ag = avg_gains[i]
        al = avg_losses[i]
        
        if al == 0:
            result.append(100.0 if ag > 0 else 0.0)
        else:
            rs = ag / al
            result.append(100 - (100 / (1 + rs)))
    
    return result


def calculate_macd(
    prices: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> tuple[list[float], list[float], list[float]]:
    """Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: List of price values
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    if len(prices) < slow or fast < 1 or slow < 1 or signal < 1:
        nan_list = [float('nan')] * len(prices)
        return nan_list, nan_list.copy(), nan_list.copy()
    
    fast_ema = calculate_ema(prices, fast)
    slow_ema = calculate_ema(prices, slow)
    
    # MACD line = Fast EMA - Slow EMA
    macd_line = []
    for f, s in zip(fast_ema, slow_ema):
        if f != f or s != s:  # Check for NaN
            macd_line.append(float('nan'))
        else:
            macd_line.append(f - s)
    
    # Signal line = EMA of MACD line
    # Only calculate EMA on valid MACD values
    valid_macd_start = slow - 1
    valid_macd = macd_line[valid_macd_start:]
    
    if len(valid_macd) < signal:
        nan_list = [float('nan')] * len(prices)
        return macd_line, nan_list, nan_list
    
    signal_ema = calculate_ema(valid_macd, signal)
    signal_line = [float('nan')] * valid_macd_start + signal_ema
    
    # Histogram = MACD - Signal
    histogram = []
    for m, s in zip(macd_line, signal_line):
        if m != m or s != s:  # Check for NaN
            histogram.append(float('nan'))
        else:
            histogram.append(m - s)
    
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    prices: list[float],
    period: int = 20,
    std_dev: float = 2.0
) -> tuple[list[float], list[float], list[float]]:
    """Calculate Bollinger Bands.
    
    Args:
        prices: List of price values
        period: SMA period (default 20)
        std_dev: Standard deviation multiplier (default 2.0)
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    if len(prices) < period or period < 1:
        nan_list = [float('nan')] * len(prices)
        return nan_list, nan_list.copy(), nan_list.copy()
    
    middle_band = calculate_sma(prices, period)
    upper_band = [float('nan')] * len(prices)
    lower_band = [float('nan')] * len(prices)
    
    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1:i + 1]
        mean = middle_band[i]
        
        # Calculate standard deviation
        variance = sum((x - mean) ** 2 for x in window) / period
        std = variance ** 0.5
        
        upper_band[i] = mean + (std_dev * std)
        lower_band[i] = mean - (std_dev * std)
    
    return upper_band, middle_band, lower_band


def calculate_atr(
    high: list[float],
    low: list[float],
    close: list[float],
    period: int = 14
) -> list[float]:
    """Calculate Average True Range.
    
    Args:
        high: List of high prices
        low: List of low prices
        close: List of close prices
        period: ATR period (default 14)
        
    Returns:
        List of ATR values
    """
    n = len(close)
    if n < period + 1 or len(high) != n or len(low) != n or period < 1:
        return [float('nan')] * n
    
    # Calculate True Range
    true_ranges = [high[0] - low[0]]  # First TR is just high - low
    
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        true_ranges.append(tr)
    
    # First ATR is SMA of first `period` true ranges
    result = [float('nan')] * (period - 1)
    first_atr = sum(true_ranges[:period]) / period
    result.append(first_atr)
    
    # Subsequent ATRs using Wilder's smoothing
    for i in range(period, n):
        atr = (result[-1] * (period - 1) + true_ranges[i]) / period
        result.append(atr)
    
    return result


def calculate_vwap(
    high: list[float],
    low: list[float],
    close: list[float],
    volume: list[float]
) -> list[float]:
    """Calculate Volume Weighted Average Price.
    
    VWAP is typically calculated from the start of the trading session.
    This implementation calculates cumulative VWAP from the start of the data.
    
    Args:
        high: List of high prices
        low: List of low prices
        close: List of close prices
        volume: List of volume values
        
    Returns:
        List of VWAP values
    """
    n = len(close)
    if n == 0 or len(high) != n or len(low) != n or len(volume) != n:
        return [float('nan')] * n
    
    result = []
    cumulative_tp_vol = 0.0
    cumulative_vol = 0.0
    
    for i in range(n):
        # Typical price = (High + Low + Close) / 3
        typical_price = (high[i] + low[i] + close[i]) / 3
        
        cumulative_tp_vol += typical_price * volume[i]
        cumulative_vol += volume[i]
        
        if cumulative_vol == 0:
            result.append(float('nan'))
        else:
            result.append(cumulative_tp_vol / cumulative_vol)
    
    return result


def calculate_stochastic(
    high: list[float],
    low: list[float],
    close: list[float],
    k_period: int = 14,
    d_period: int = 3
) -> tuple[list[float], list[float]]:
    """Calculate Stochastic Oscillator (%K and %D).
    
    Args:
        high: List of high prices
        low: List of low prices
        close: List of close prices
        k_period: %K period (default 14)
        d_period: %D smoothing period (default 3)
        
    Returns:
        Tuple of (%K values, %D values)
    """
    n = len(close)
    if n < k_period or len(high) != n or len(low) != n:
        nan_list = [float('nan')] * n
        return nan_list, nan_list.copy()
    
    k_values = [float('nan')] * (k_period - 1)
    
    for i in range(k_period - 1, n):
        highest_high = max(high[i - k_period + 1:i + 1])
        lowest_low = min(low[i - k_period + 1:i + 1])
        
        if highest_high == lowest_low:
            k_values.append(50.0)  # Neutral when no range
        else:
            k = ((close[i] - lowest_low) / (highest_high - lowest_low)) * 100
            k_values.append(k)
    
    # %D is SMA of %K
    d_values = calculate_sma(k_values, d_period)
    
    return k_values, d_values


def calculate_adx(
    high: list[float],
    low: list[float],
    close: list[float],
    period: int = 14
) -> tuple[list[float], list[float], list[float]]:
    """Calculate Average Directional Index (ADX) with +DI and -DI.
    
    Args:
        high: List of high prices
        low: List of low prices
        close: List of close prices
        period: ADX period (default 14)
        
    Returns:
        Tuple of (ADX, +DI, -DI)
    """
    n = len(close)
    if n < period * 2 or len(high) != n or len(low) != n:
        nan_list = [float('nan')] * n
        return nan_list, nan_list.copy(), nan_list.copy()
    
    # Calculate +DM and -DM
    plus_dm = [0.0]
    minus_dm = [0.0]
    tr_list = [high[0] - low[0]]
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0.0)
            
        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0.0)
        
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        tr_list.append(tr)
    
    # Smooth using Wilder's method
    def wilder_smooth(data: list[float], period: int) -> list[float]:
        result = [float('nan')] * (period - 1)
        first_sum = sum(data[:period])
        result.append(first_sum)
        for i in range(period, len(data)):
            smoothed = result[-1] - (result[-1] / period) + data[i]
            result.append(smoothed)
        return result
    
    smoothed_tr = wilder_smooth(tr_list, period)
    smoothed_plus_dm = wilder_smooth(plus_dm, period)
    smoothed_minus_dm = wilder_smooth(minus_dm, period)
    
    # Calculate +DI and -DI
    plus_di = [float('nan')] * n
    minus_di = [float('nan')] * n
    dx = [float('nan')] * n
    
    for i in range(period - 1, n):
        if smoothed_tr[i] > 0:
            plus_di[i] = (smoothed_plus_dm[i] / smoothed_tr[i]) * 100
            minus_di[i] = (smoothed_minus_dm[i] / smoothed_tr[i]) * 100
            
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = abs(plus_di[i] - minus_di[i]) / di_sum * 100
    
    # ADX is smoothed DX
    adx = [float('nan')] * n
    valid_dx = [d for d in dx if d == d]  # Filter NaN
    
    if len(valid_dx) >= period:
        first_adx_idx = period * 2 - 2
        adx[first_adx_idx] = sum(dx[period - 1:first_adx_idx + 1]) / period
        
        for i in range(first_adx_idx + 1, n):
            if dx[i] == dx[i]:  # Not NaN
                adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx, plus_di, minus_di


def calculate_supertrend(
    high: list[float],
    low: list[float],
    close: list[float],
    period: int = 10,
    multiplier: float = 3.0
) -> tuple[list[float], list[int]]:
    """Calculate SuperTrend indicator.
    
    Args:
        high: List of high prices
        low: List of low prices
        close: List of close prices
        period: ATR period (default 10)
        multiplier: ATR multiplier (default 3.0)
        
    Returns:
        Tuple of (SuperTrend values, Direction: 1=bullish, -1=bearish)
    """
    n = len(close)
    if n < period + 1 or len(high) != n or len(low) != n:
        return [float('nan')] * n, [0] * n
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = [float('nan')] * n
    direction = [0] * n
    
    upper_band = [float('nan')] * n
    lower_band = [float('nan')] * n
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2
        upper_band[i] = hl2 + (multiplier * atr[i])
        lower_band[i] = hl2 - (multiplier * atr[i])
        
        if i == period:
            supertrend[i] = upper_band[i]
            direction[i] = -1
        else:
            # Adjust bands based on previous values
            if lower_band[i] > lower_band[i - 1] or close[i - 1] < lower_band[i - 1]:
                pass  # Keep current lower band
            else:
                lower_band[i] = lower_band[i - 1]
                
            if upper_band[i] < upper_band[i - 1] or close[i - 1] > upper_band[i - 1]:
                pass  # Keep current upper band
            else:
                upper_band[i] = upper_band[i - 1]
            
            # Determine trend direction
            if supertrend[i - 1] == upper_band[i - 1]:
                if close[i] > upper_band[i]:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1
                else:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
            else:
                if close[i] < lower_band[i]:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
                else:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1
    
    return supertrend, direction


def calculate_fibonacci_levels(
    high_price: float,
    low_price: float,
    trend: str = "up"
) -> dict[str, float]:
    """Calculate Fibonacci retracement levels.
    
    Args:
        high_price: Swing high price
        low_price: Swing low price
        trend: "up" for uptrend (retracement from high), "down" for downtrend
        
    Returns:
        Dictionary with Fibonacci levels (0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%)
    """
    diff = high_price - low_price
    
    levels = {
        "0.0%": high_price if trend == "up" else low_price,
        "23.6%": high_price - (0.236 * diff) if trend == "up" else low_price + (0.236 * diff),
        "38.2%": high_price - (0.382 * diff) if trend == "up" else low_price + (0.382 * diff),
        "50.0%": high_price - (0.5 * diff) if trend == "up" else low_price + (0.5 * diff),
        "61.8%": high_price - (0.618 * diff) if trend == "up" else low_price + (0.618 * diff),
        "78.6%": high_price - (0.786 * diff) if trend == "up" else low_price + (0.786 * diff),
        "100.0%": low_price if trend == "up" else high_price,
    }
    
    return levels


def calculate_pivot_points(
    high: float,
    low: float,
    close: float
) -> dict[str, float]:
    """Calculate Standard Pivot Points with support and resistance levels.
    
    Args:
        high: Previous day's high
        low: Previous day's low
        close: Previous day's close
        
    Returns:
        Dictionary with Pivot, R1-R3, S1-S3 levels
    """
    pivot = (high + low + close) / 3
    
    return {
        "R3": high + 2 * (pivot - low),
        "R2": pivot + (high - low),
        "R1": (2 * pivot) - low,
        "Pivot": pivot,
        "S1": (2 * pivot) - high,
        "S2": pivot - (high - low),
        "S3": low - 2 * (high - pivot),
    }


def calculate_obv(close: list[float], volume: list[float]) -> list[float]:
    """Calculate On-Balance Volume (OBV).
    
    Args:
        close: List of close prices
        volume: List of volume values
        
    Returns:
        List of OBV values
    """
    n = len(close)
    if n == 0 or len(volume) != n:
        return [float('nan')] * n
    
    obv = [volume[0]]
    
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv.append(obv[-1] + volume[i])
        elif close[i] < close[i - 1]:
            obv.append(obv[-1] - volume[i])
        else:
            obv.append(obv[-1])
    
    return obv


def calculate_cci(
    high: list[float],
    low: list[float],
    close: list[float],
    period: int = 20
) -> list[float]:
    """Calculate Commodity Channel Index (CCI).
    
    Args:
        high: List of high prices
        low: List of low prices
        close: List of close prices
        period: CCI period (default 20)
        
    Returns:
        List of CCI values
    """
    n = len(close)
    if n < period or len(high) != n or len(low) != n:
        return [float('nan')] * n
    
    # Typical Price
    tp = [(h + l + c) / 3 for h, l, c in zip(high, low, close)]
    
    # SMA of Typical Price
    tp_sma = calculate_sma(tp, period)
    
    result = [float('nan')] * (period - 1)
    
    for i in range(period - 1, n):
        window = tp[i - period + 1:i + 1]
        mean = tp_sma[i]
        
        # Mean Deviation
        mean_dev = sum(abs(x - mean) for x in window) / period
        
        if mean_dev == 0:
            result.append(0.0)
        else:
            cci = (tp[i] - mean) / (0.015 * mean_dev)
            result.append(cci)
    
    return result


def calculate_williams_r(
    high: list[float],
    low: list[float],
    close: list[float],
    period: int = 14
) -> list[float]:
    """Calculate Williams %R.
    
    Args:
        high: List of high prices
        low: List of low prices
        close: List of close prices
        period: Lookback period (default 14)
        
    Returns:
        List of Williams %R values (-100 to 0)
    """
    n = len(close)
    if n < period or len(high) != n or len(low) != n:
        return [float('nan')] * n
    
    result = [float('nan')] * (period - 1)
    
    for i in range(period - 1, n):
        highest_high = max(high[i - period + 1:i + 1])
        lowest_low = min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            result.append(-50.0)
        else:
            wr = ((highest_high - close[i]) / (highest_high - lowest_low)) * -100
            result.append(wr)
    
    return result


def detect_candlestick_patterns(
    open_prices: list[float],
    high: list[float],
    low: list[float],
    close: list[float]
) -> list[dict]:
    """Detect common candlestick patterns.
    
    Args:
        open_prices: List of open prices
        high: List of high prices
        low: List of low prices
        close: List of close prices
        
    Returns:
        List of detected patterns with index and pattern name
    """
    n = len(close)
    if n < 3 or len(open_prices) != n or len(high) != n or len(low) != n:
        return []
    
    patterns = []
    
    for i in range(2, n):
        o, h, l, c = open_prices[i], high[i], low[i], close[i]
        body = abs(c - o)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        total_range = h - l
        
        if total_range == 0:
            continue
        
        # Doji - small body
        if body / total_range < 0.1:
            patterns.append({"index": i, "pattern": "Doji", "signal": "neutral"})
        
        # Hammer - small body at top, long lower shadow
        elif lower_shadow > body * 2 and upper_shadow < body * 0.5 and c > o:
            patterns.append({"index": i, "pattern": "Hammer", "signal": "bullish"})
        
        # Inverted Hammer
        elif upper_shadow > body * 2 and lower_shadow < body * 0.5 and c > o:
            patterns.append({"index": i, "pattern": "Inverted Hammer", "signal": "bullish"})
        
        # Shooting Star
        elif upper_shadow > body * 2 and lower_shadow < body * 0.5 and c < o:
            patterns.append({"index": i, "pattern": "Shooting Star", "signal": "bearish"})
        
        # Hanging Man
        elif lower_shadow > body * 2 and upper_shadow < body * 0.5 and c < o:
            patterns.append({"index": i, "pattern": "Hanging Man", "signal": "bearish"})
        
        # Engulfing patterns (need previous candle)
        prev_o, prev_c = open_prices[i - 1], close[i - 1]
        prev_body = abs(prev_c - prev_o)
        
        # Bullish Engulfing
        if prev_c < prev_o and c > o and o < prev_c and c > prev_o and body > prev_body:
            patterns.append({"index": i, "pattern": "Bullish Engulfing", "signal": "bullish"})
        
        # Bearish Engulfing
        elif prev_c > prev_o and c < o and o > prev_c and c < prev_o and body > prev_body:
            patterns.append({"index": i, "pattern": "Bearish Engulfing", "signal": "bearish"})
        
        # Morning Star (3-candle pattern)
        if i >= 2:
            c2_o, c2_c = open_prices[i - 2], close[i - 2]
            c1_o, c1_c, c1_body = prev_o, prev_c, prev_body
            c1_range = high[i - 1] - low[i - 1]
            
            # First candle bearish, second small body (star), third bullish
            if (c2_c < c2_o and  # First bearish
                c1_body < c1_range * 0.3 and  # Second small body (star)
                c > o and  # Third bullish
                c > (c2_o + c2_c) / 2):  # Third closes above midpoint of first
                patterns.append({"index": i, "pattern": "Morning Star", "signal": "bullish"})
            
            # Evening Star (opposite of Morning Star)
            elif (c2_c > c2_o and  # First bullish
                  c1_body < c1_range * 0.3 and  # Second small body (star)
                  c < o and  # Third bearish
                  c < (c2_o + c2_c) / 2):  # Third closes below midpoint of first
                patterns.append({"index": i, "pattern": "Evening Star", "signal": "bearish"})
        
        # Three White Soldiers (3 consecutive bullish candles)
        if i >= 2:
            c2_o, c2_c = open_prices[i - 2], close[i - 2]
            c1_o, c1_c = prev_o, prev_c
            if (c2_c > c2_o and c1_c > c1_o and c > o and  # All bullish
                c1_c > c2_c and c > c1_c and  # Each closes higher
                c1_o > c2_o and o > c1_o):  # Each opens higher
                patterns.append({"index": i, "pattern": "Three White Soldiers", "signal": "bullish"})
        
        # Three Black Crows (3 consecutive bearish candles)
        if i >= 2:
            c2_o, c2_c = open_prices[i - 2], close[i - 2]
            c1_o, c1_c = prev_o, prev_c
            if (c2_c < c2_o and c1_c < c1_o and c < o and  # All bearish
                c1_c < c2_c and c < c1_c and  # Each closes lower
                c1_o < c2_o and o < c1_o):  # Each opens lower
                patterns.append({"index": i, "pattern": "Three Black Crows", "signal": "bearish"})
        
        # Marubozu (strong candle with no/tiny shadows)
        if upper_shadow < body * 0.1 and lower_shadow < body * 0.1:
            if c > o:
                patterns.append({"index": i, "pattern": "Bullish Marubozu", "signal": "bullish"})
            else:
                patterns.append({"index": i, "pattern": "Bearish Marubozu", "signal": "bearish"})
    
    return patterns


def calculate_signal_score(
    high: list[float],
    low: list[float],
    close: list[float],
    volume: list[float],
) -> dict:
    """Calculate a composite signal score from multiple indicators.
    
    Combines RSI, MACD, Stochastic, SuperTrend, ADX, and moving averages
    into a single score from -100 (strong sell) to +100 (strong buy).
    
    Args:
        high: List of high prices
        low: List of low prices
        close: List of close prices
        volume: List of volume values
        
    Returns:
        Dictionary with total score, individual scores, and recommendation
    """
    if len(close) < 50:
        return {"error": "Insufficient data (need 50+ candles)"}
    
    scores = {}
    weights = {}
    
    # RSI Score (-20 to +20)
    rsi_values = calculate_rsi(close, 14)
    if rsi_values and rsi_values[-1] == rsi_values[-1]:
        rsi = rsi_values[-1]
        if rsi < 30:
            scores["rsi"] = 20  # Oversold = bullish
        elif rsi < 40:
            scores["rsi"] = 10
        elif rsi > 70:
            scores["rsi"] = -20  # Overbought = bearish
        elif rsi > 60:
            scores["rsi"] = -10
        else:
            scores["rsi"] = 0
        weights["rsi"] = 1.0
    
    # MACD Score (-15 to +15)
    macd_line, signal_line, histogram = calculate_macd(close)
    if histogram and histogram[-1] == histogram[-1]:
        hist = histogram[-1]
        prev_hist = histogram[-2] if len(histogram) > 1 else 0
        
        if hist > 0 and hist > prev_hist:
            scores["macd"] = 15  # Bullish and increasing
        elif hist > 0:
            scores["macd"] = 8
        elif hist < 0 and hist < prev_hist:
            scores["macd"] = -15  # Bearish and decreasing
        elif hist < 0:
            scores["macd"] = -8
        else:
            scores["macd"] = 0
        weights["macd"] = 1.0
    
    # Stochastic Score (-15 to +15)
    k_values, d_values = calculate_stochastic(high, low, close)
    if k_values and k_values[-1] == k_values[-1]:
        k, d = k_values[-1], d_values[-1]
        if k < 20:
            scores["stoch"] = 15  # Oversold
        elif k < 30 and k > d:
            scores["stoch"] = 10  # Bullish crossover in oversold
        elif k > 80:
            scores["stoch"] = -15  # Overbought
        elif k > 70 and k < d:
            scores["stoch"] = -10  # Bearish crossover in overbought
        elif k > d:
            scores["stoch"] = 5
        elif k < d:
            scores["stoch"] = -5
        else:
            scores["stoch"] = 0
        weights["stoch"] = 1.0
    
    # SuperTrend Score (-20 to +20)
    st_values, st_direction = calculate_supertrend(high, low, close)
    if st_direction and st_direction[-1] != 0:
        direction = st_direction[-1]
        prev_direction = st_direction[-2] if len(st_direction) > 1 else direction
        
        if direction == 1 and prev_direction == -1:
            scores["supertrend"] = 20  # Fresh bullish signal
        elif direction == 1:
            scores["supertrend"] = 12
        elif direction == -1 and prev_direction == 1:
            scores["supertrend"] = -20  # Fresh bearish signal
        elif direction == -1:
            scores["supertrend"] = -12
        weights["supertrend"] = 1.2  # Higher weight for trend
    
    # ADX Trend Strength (modifies other scores)
    adx_values, plus_di, minus_di = calculate_adx(high, low, close)
    if adx_values and adx_values[-1] == adx_values[-1]:
        adx = adx_values[-1]
        pdi, mdi = plus_di[-1], minus_di[-1]
        
        if adx > 25:
            if pdi > mdi:
                scores["adx"] = 10  # Strong bullish trend
            else:
                scores["adx"] = -10  # Strong bearish trend
        elif adx < 20:
            scores["adx"] = 0  # No clear trend
        else:
            scores["adx"] = 5 if pdi > mdi else -5
        weights["adx"] = 0.8
    
    # Moving Average Score (-15 to +15)
    ema_9 = calculate_ema(close, 9)
    ema_21 = calculate_ema(close, 21)
    sma_50 = calculate_sma(close, 50)
    
    if ema_9 and ema_21 and ema_9[-1] == ema_9[-1]:
        price = close[-1]
        ma_score = 0
        
        if price > ema_9[-1]:
            ma_score += 5
        else:
            ma_score -= 5
            
        if ema_9[-1] > ema_21[-1]:
            ma_score += 5
        else:
            ma_score -= 5
            
        if sma_50[-1] == sma_50[-1] and price > sma_50[-1]:
            ma_score += 5
        elif sma_50[-1] == sma_50[-1]:
            ma_score -= 5
            
        scores["ma"] = ma_score
        weights["ma"] = 1.0
    
    # Volume confirmation
    if len(volume) >= 20:
        avg_vol = sum(volume[-20:]) / 20
        current_vol = volume[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
        
        # High volume confirms the move
        if vol_ratio > 1.5:
            scores["volume"] = 5 if close[-1] > close[-2] else -5
        else:
            scores["volume"] = 0
        weights["volume"] = 0.5
    
    # Calculate weighted total
    total_weight = sum(weights.values())
    weighted_sum = sum(scores.get(k, 0) * weights.get(k, 1) for k in scores)
    
    # Normalize to -100 to +100 scale
    max_possible = 20 + 15 + 15 + 20 + 10 + 15 + 5  # Sum of max positive scores
    normalized_score = (weighted_sum / total_weight) * (100 / max_possible) * total_weight
    normalized_score = max(-100, min(100, normalized_score))
    
    # Generate recommendation
    if normalized_score >= 50:
        recommendation = "STRONG BUY"
        color = "green"
    elif normalized_score >= 25:
        recommendation = "BUY"
        color = "green"
    elif normalized_score >= 10:
        recommendation = "WEAK BUY"
        color = "yellow"
    elif normalized_score <= -50:
        recommendation = "STRONG SELL"
        color = "red"
    elif normalized_score <= -25:
        recommendation = "SELL"
        color = "red"
    elif normalized_score <= -10:
        recommendation = "WEAK SELL"
        color = "yellow"
    else:
        recommendation = "NEUTRAL"
        color = "dim"
    
    return {
        "total_score": round(normalized_score, 1),
        "recommendation": recommendation,
        "color": color,
        "breakdown": scores,
        "weights": weights,
    }
