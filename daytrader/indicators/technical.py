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
