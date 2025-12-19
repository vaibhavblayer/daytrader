"""Analyze command for DayTrader CLI.

Calculates and displays technical indicators for stock analysis.
"""

from datetime import date, timedelta
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from daytrader.indicators.technical import (
    calculate_rsi,
    calculate_macd,
    calculate_ema,
    calculate_sma,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_vwap,
    calculate_stochastic,
    calculate_adx,
    calculate_supertrend,
    calculate_fibonacci_levels,
    calculate_pivot_points,
    calculate_obv,
    calculate_cci,
    calculate_williams_r,
    detect_candlestick_patterns,
    calculate_signal_score,
)

console = Console()


def _get_market_status() -> tuple[str, str, bool]:
    """Get current market status.
    
    Returns:
        Tuple of (status_text, color, is_open)
    """
    from datetime import datetime
    import pytz
    
    # IST timezone
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    
    # Market hours: 9:15 AM to 3:30 PM, Monday to Friday
    # MIS auto square-off: 3:15-3:20 PM
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    mis_squareoff = now.replace(hour=15, minute=15, second=0, microsecond=0)
    pre_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    
    is_weekday = now.weekday() < 5  # Monday = 0, Friday = 4
    
    if not is_weekday:
        return "CLOSED (Weekend)", "red", False
    elif now < pre_open:
        return "CLOSED (Pre-market)", "yellow", False
    elif now < market_open:
        return "PRE-OPEN (9:00-9:15)", "yellow", False
    elif now > market_close:
        return "CLOSED (After hours)", "red", False
    elif now >= mis_squareoff:
        # After 3:15 PM - MIS positions auto square-off
        mins_left = (market_close - now).seconds // 60
        return f"MIS CLOSED - Auto square-off ({mins_left} mins to close)", "red", False
    elif now >= mis_squareoff.replace(minute=0):  # After 3:00 PM
        mins_to_squareoff = (mis_squareoff - now).seconds // 60
        return f"⚠ MIS CLOSING ({mins_to_squareoff} mins to auto square-off)", "yellow", True
    else:
        return "OPEN", "green", True


# Available indicators
AVAILABLE_INDICATORS = [
    "rsi", "macd", "ema", "sma", "bb", "atr", "vwap",
    "stoch", "adx", "supertrend", "fib", "pivot", "obv", "cci", "willr", "patterns"
]
DEFAULT_INDICATORS = ["bb", "supertrend", "fib", "rsi", "adx"]


def _get_config():
    """Lazily load configuration."""
    from pathlib import Path
    import toml
    
    config_path = Path.home() / ".config" / "daytrader" / "config.toml"
    
    if not config_path.exists():
        return None
    
    try:
        return toml.load(config_path)
    except Exception:
        return None


def _get_data_store():
    """Get the data store instance."""
    from pathlib import Path
    from daytrader.db.store import DataStore
    
    db_path = Path.home() / ".config" / "daytrader" / "daytrader.db"
    return DataStore(db_path)


def _get_ai_analysis(symbol: str, timeframe: str, results: dict, config: dict) -> str:
    """Get AI verification and trading advice using the Technical Verification Agent.
    
    Args:
        symbol: Trading symbol.
        timeframe: Candle timeframe.
        results: Indicator results dictionary.
        config: Application config with OpenAI settings.
        
    Returns:
        AI analysis text.
    """
    try:
        from daytrader.agents.technical import get_ai_verification
    except ImportError as e:
        return f"[yellow]Agent SDK not available: {e}[/yellow]"
    
    # Get model from config (default: o4-mini for medium reasoning)
    openai_config = config.get("openai", {})
    model = openai_config.get("model", "o4-mini")
    
    try:
        return get_ai_verification(
            symbol=symbol,
            timeframe=timeframe,
            results=results,
            model=model,
        )
    except Exception as e:
        return f"[red]AI Error: {str(e)}[/red]"


def _parse_indicators(indicators_str: str) -> list[str]:
    """Parse comma-separated indicator string into list.
    
    Args:
        indicators_str: Comma-separated indicator names.
        
    Returns:
        List of valid indicator names.
    """
    requested = [ind.strip().lower() for ind in indicators_str.split(",")]
    valid = [ind for ind in requested if ind in AVAILABLE_INDICATORS]
    return valid if valid else DEFAULT_INDICATORS


def _interpret_rsi(value: float) -> tuple[str, str]:
    """Interpret RSI value and return signal and color.
    
    Args:
        value: RSI value (0-100).
        
    Returns:
        Tuple of (signal_text, color).
    """
    if value < 30:
        return "Oversold (Buy Signal)", "green"
    elif value > 70:
        return "Overbought (Sell Signal)", "red"
    elif value < 40:
        return "Approaching Oversold", "yellow"
    elif value > 60:
        return "Approaching Overbought", "yellow"
    else:
        return "Neutral", "dim"


def _interpret_macd(macd: float, signal: float, histogram: float) -> tuple[str, str]:
    """Interpret MACD values and return signal and color.
    
    Args:
        macd: MACD line value.
        signal: Signal line value.
        histogram: Histogram value.
        
    Returns:
        Tuple of (signal_text, color).
    """
    if histogram > 0 and macd > signal:
        return "Bullish (Buy Signal)", "green"
    elif histogram < 0 and macd < signal:
        return "Bearish (Sell Signal)", "red"
    elif histogram > 0:
        return "Bullish Momentum", "green"
    elif histogram < 0:
        return "Bearish Momentum", "red"
    else:
        return "Neutral", "dim"


def _interpret_bb(price: float, upper: float, middle: float, lower: float) -> tuple[str, str]:
    """Interpret Bollinger Bands position and return signal and color.
    
    Args:
        price: Current price.
        upper: Upper band.
        middle: Middle band (SMA).
        lower: Lower band.
        
    Returns:
        Tuple of (signal_text, color).
    """
    if price <= lower:
        return "At Lower Band (Potential Buy)", "green"
    elif price >= upper:
        return "At Upper Band (Potential Sell)", "red"
    elif price < middle:
        return "Below Middle Band", "yellow"
    elif price > middle:
        return "Above Middle Band", "yellow"
    else:
        return "At Middle Band", "dim"


def calculate_indicators_for_symbol(
    symbol: str,
    indicators: list[str],
    store,
    days: int = 100,
    timeframe: str = "1day",
) -> dict:
    """Calculate requested indicators for a symbol.
    
    Args:
        symbol: Trading symbol.
        indicators: List of indicator names to calculate.
        store: DataStore instance.
        days: Number of days of data to use.
        timeframe: Candle interval (1min, 5min, 15min, 1hour, 1day).
        
    Returns:
        Dictionary with indicator results.
    """
    # Get candle data from cache
    to_date = date.today()
    from_date = to_date - timedelta(days=days)
    
    candles = store.get_candles(symbol, timeframe, from_date, to_date)
    
    if not candles or len(candles) < 30:
        return {"error": f"Insufficient data for {symbol}. Need at least 30 candles."}
    
    # Extract price arrays
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [float(c.volume) for c in candles]
    
    results = {
        "symbol": symbol,
        "last_price": closes[-1],
        "candle_count": len(candles),
        "indicators": {},
    }
    
    # Calculate only requested indicators
    if "rsi" in indicators:
        rsi_values = calculate_rsi(closes, period=14)
        latest_rsi = rsi_values[-1] if rsi_values else float('nan')
        if latest_rsi == latest_rsi:  # Check for NaN
            signal, color = _interpret_rsi(latest_rsi)
            results["indicators"]["rsi"] = {
                "value": latest_rsi,
                "signal": signal,
                "color": color,
            }
    
    if "macd" in indicators:
        macd_line, signal_line, histogram = calculate_macd(closes)
        if macd_line and macd_line[-1] == macd_line[-1]:  # Check for NaN
            signal, color = _interpret_macd(
                macd_line[-1], signal_line[-1], histogram[-1]
            )
            results["indicators"]["macd"] = {
                "macd": macd_line[-1],
                "signal": signal_line[-1],
                "histogram": histogram[-1],
                "interpretation": signal,
                "color": color,
            }
    
    if "ema" in indicators:
        ema_9 = calculate_ema(closes, 9)
        ema_21 = calculate_ema(closes, 21)
        if ema_9 and ema_9[-1] == ema_9[-1]:
            trend = "Bullish" if ema_9[-1] > ema_21[-1] else "Bearish"
            color = "green" if trend == "Bullish" else "red"
            results["indicators"]["ema"] = {
                "ema_9": ema_9[-1],
                "ema_21": ema_21[-1],
                "trend": trend,
                "color": color,
            }
    
    if "sma" in indicators:
        sma_20 = calculate_sma(closes, 20)
        sma_50 = calculate_sma(closes, 50)
        if sma_20 and sma_20[-1] == sma_20[-1]:
            trend = "Bullish" if closes[-1] > sma_20[-1] else "Bearish"
            color = "green" if trend == "Bullish" else "red"
            results["indicators"]["sma"] = {
                "sma_20": sma_20[-1],
                "sma_50": sma_50[-1] if sma_50[-1] == sma_50[-1] else None,
                "trend": trend,
                "color": color,
            }
    
    if "bb" in indicators:
        upper, middle, lower = calculate_bollinger_bands(closes)
        if upper and upper[-1] == upper[-1]:
            signal, color = _interpret_bb(closes[-1], upper[-1], middle[-1], lower[-1])
            results["indicators"]["bb"] = {
                "upper": upper[-1],
                "middle": middle[-1],
                "lower": lower[-1],
                "signal": signal,
                "color": color,
            }
    
    if "atr" in indicators:
        atr_values = calculate_atr(highs, lows, closes)
        if atr_values and atr_values[-1] == atr_values[-1]:
            atr_pct = (atr_values[-1] / closes[-1]) * 100
            volatility = "High" if atr_pct > 3 else "Low" if atr_pct < 1 else "Normal"
            color = "red" if volatility == "High" else "green" if volatility == "Low" else "yellow"
            results["indicators"]["atr"] = {
                "value": atr_values[-1],
                "percent": atr_pct,
                "volatility": volatility,
                "color": color,
            }
    
    if "vwap" in indicators:
        vwap_values = calculate_vwap(highs, lows, closes, volumes)
        if vwap_values and vwap_values[-1] == vwap_values[-1]:
            position = "Above VWAP" if closes[-1] > vwap_values[-1] else "Below VWAP"
            color = "green" if position == "Above VWAP" else "red"
            results["indicators"]["vwap"] = {
                "value": vwap_values[-1],
                "position": position,
                "color": color,
            }
    
    if "stoch" in indicators:
        k_values, d_values = calculate_stochastic(highs, lows, closes)
        if k_values and k_values[-1] == k_values[-1]:
            k, d = k_values[-1], d_values[-1]
            if k < 20:
                signal, color = "Oversold (Buy Signal)", "green"
            elif k > 80:
                signal, color = "Overbought (Sell Signal)", "red"
            elif k > d:
                signal, color = "Bullish Crossover", "green"
            elif k < d:
                signal, color = "Bearish Crossover", "red"
            else:
                signal, color = "Neutral", "dim"
            results["indicators"]["stoch"] = {
                "k": k,
                "d": d,
                "signal": signal,
                "color": color,
            }
    
    if "adx" in indicators:
        adx_values, plus_di, minus_di = calculate_adx(highs, lows, closes)
        if adx_values and adx_values[-1] == adx_values[-1]:
            adx = adx_values[-1]
            pdi, mdi = plus_di[-1], minus_di[-1]
            if adx < 20:
                trend_strength = "Weak/No Trend"
                color = "dim"
            elif adx < 40:
                trend_strength = "Moderate Trend"
                color = "yellow"
            else:
                trend_strength = "Strong Trend"
                color = "green" if pdi > mdi else "red"
            direction = "Bullish" if pdi > mdi else "Bearish"
            results["indicators"]["adx"] = {
                "adx": adx,
                "plus_di": pdi,
                "minus_di": mdi,
                "trend_strength": trend_strength,
                "direction": direction,
                "color": color,
            }
    
    if "supertrend" in indicators:
        st_values, st_direction = calculate_supertrend(highs, lows, closes)
        if st_values and st_values[-1] == st_values[-1]:
            direction = st_direction[-1]
            signal = "Bullish (Buy)" if direction == 1 else "Bearish (Sell)"
            color = "green" if direction == 1 else "red"
            results["indicators"]["supertrend"] = {
                "value": st_values[-1],
                "direction": direction,
                "signal": signal,
                "color": color,
            }
    
    if "fib" in indicators:
        # Find recent swing high and low (last 20 candles)
        recent_high = max(highs[-20:])
        recent_low = min(lows[-20:])
        trend = "up" if closes[-1] > closes[-20] else "down"
        fib_levels = calculate_fibonacci_levels(recent_high, recent_low, trend)
        
        # Find nearest support and resistance levels
        price = closes[-1]
        sorted_levels = sorted(fib_levels.items(), key=lambda x: x[1])
        nearest_support = None
        nearest_resistance = None
        current_zone = None
        
        for i, (level, level_price) in enumerate(sorted_levels):
            if level_price < price:
                nearest_support = (level, level_price)
            elif level_price >= price and nearest_resistance is None:
                nearest_resistance = (level, level_price)
        
        # Determine which zone price is in
        for i in range(len(sorted_levels) - 1):
            low_level, low_price = sorted_levels[i]
            high_level, high_price = sorted_levels[i + 1]
            if low_price <= price <= high_price:
                current_zone = f"{low_level} - {high_level}"
                break
        
        results["indicators"]["fib"] = {
            "levels": fib_levels,
            "trend": trend,
            "swing_high": recent_high,
            "swing_low": recent_low,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "current_zone": current_zone,
            "price": price,
        }
    
    if "pivot" in indicators:
        # Use previous day's data
        if len(candles) >= 2:
            prev = candles[-2]
            pivot_levels = calculate_pivot_points(prev.high, prev.low, prev.close)
            # Determine current position
            price = closes[-1]
            if price > pivot_levels["R1"]:
                position = "Above R1 (Bullish)"
                color = "green"
            elif price < pivot_levels["S1"]:
                position = "Below S1 (Bearish)"
                color = "red"
            else:
                position = "Between S1 and R1"
                color = "yellow"
            results["indicators"]["pivot"] = {
                "levels": pivot_levels,
                "position": position,
                "color": color,
            }
    
    if "obv" in indicators:
        obv_values = calculate_obv(closes, volumes)
        if obv_values and len(obv_values) >= 5:
            obv_trend = "Rising" if obv_values[-1] > obv_values[-5] else "Falling"
            color = "green" if obv_trend == "Rising" else "red"
            results["indicators"]["obv"] = {
                "value": obv_values[-1],
                "trend": obv_trend,
                "color": color,
            }
    
    if "cci" in indicators:
        cci_values = calculate_cci(highs, lows, closes)
        if cci_values and cci_values[-1] == cci_values[-1]:
            cci = cci_values[-1]
            if cci > 100:
                signal, color = "Overbought", "red"
            elif cci < -100:
                signal, color = "Oversold", "green"
            else:
                signal, color = "Neutral", "dim"
            results["indicators"]["cci"] = {
                "value": cci,
                "signal": signal,
                "color": color,
            }
    
    if "willr" in indicators:
        willr_values = calculate_williams_r(highs, lows, closes)
        if willr_values and willr_values[-1] == willr_values[-1]:
            wr = willr_values[-1]
            if wr > -20:
                signal, color = "Overbought", "red"
            elif wr < -80:
                signal, color = "Oversold", "green"
            else:
                signal, color = "Neutral", "dim"
            results["indicators"]["willr"] = {
                "value": wr,
                "signal": signal,
                "color": color,
            }
    
    if "patterns" in indicators:
        opens = [c.open for c in candles]
        patterns = detect_candlestick_patterns(opens, highs, lows, closes)
        # Get last 3 patterns
        recent_patterns = patterns[-3:] if patterns else []
        results["indicators"]["patterns"] = {
            "detected": recent_patterns,
            "count": len(patterns),
        }
    
    return results


@click.command()
@click.argument("symbol")
@click.option(
    "--indicators",
    "-i",
    default=None,
    help=f"Comma-separated indicators. Available: {', '.join(AVAILABLE_INDICATORS)}",
)
@click.option(
    "--timeframe",
    "-t",
    default="1day",
    type=click.Choice(["1min", "5min", "15min", "1hour", "1day"]),
    help="Candle timeframe for analysis (default: 1day)",
)
@click.option(
    "--ai",
    is_flag=True,
    default=False,
    help="Get AI verification and trading advice from GPT",
)
def analyze(symbol: str, indicators: str, timeframe: str, ai: bool) -> None:
    """Calculate and display technical indicators for a symbol.
    
    SYMBOL is the trading symbol (e.g., RELIANCE, INFY, TCS).
    
    \b
    Available indicators:
      rsi        - Relative Strength Index (14-period)
      macd       - Moving Average Convergence Divergence
      ema        - Exponential Moving Averages (9, 21)
      sma        - Simple Moving Averages (20, 50)
      bb         - Bollinger Bands (20-period, 2 std dev)
      atr        - Average True Range (14-period)
      vwap       - Volume Weighted Average Price
      stoch      - Stochastic Oscillator (%K, %D)
      adx        - Average Directional Index (+DI, -DI)
      supertrend - SuperTrend indicator
      fib        - Fibonacci Retracement levels
      pivot      - Pivot Points (R1-R3, S1-S3)
      obv        - On-Balance Volume
      cci        - Commodity Channel Index
      willr      - Williams %R
      patterns   - Candlestick pattern detection
    
    \b
    Timeframes: 1min, 5min, 15min, 1hour, 1day
    
    \b
    Examples:
      daytrader analyze RELIANCE                       # Default (daily)
      daytrader analyze RELIANCE -t 5min              # 5-minute candles
      daytrader analyze RELIANCE --ai                 # With AI verification
      daytrader analyze INFY -i patterns -t 15min     # 15min patterns
    """
    config = _get_config()
    
    if config is None:
        console.print(Panel(
            "[red]Configuration not found.[/red]\n\n"
            "Run [cyan]daytrader login[/cyan] to create a config file.",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    symbol = symbol.upper()
    indicator_list = _parse_indicators(indicators) if indicators else DEFAULT_INDICATORS
    
    tf_display = {"1min": "1-minute", "5min": "5-minute", "15min": "15-minute", "1hour": "1-hour", "1day": "daily"}
    console.print(f"[dim]Analyzing {symbol} ({tf_display[timeframe]}) with indicators: {', '.join(indicator_list)}...[/dim]")
    
    store = _get_data_store()
    
    # For intraday timeframes, use fewer days but need data first
    days = 100 if timeframe == "1day" else 5
    results = calculate_indicators_for_symbol(symbol, indicator_list, store, days=days, timeframe=timeframe)
    
    if "error" in results:
        # Try to auto-fetch data
        console.print(f"[dim]No cached data, fetching {timeframe} data for {symbol}...[/dim]")
        try:
            from daytrader.brokers.angelone import AngelOneBroker
            
            angelone_config = config.get("angelone", {})
            if angelone_config.get("api_key"):
                broker = AngelOneBroker(
                    api_key=angelone_config.get("api_key", ""),
                    client_id=angelone_config.get("client_id", ""),
                    pin=angelone_config.get("pin", ""),
                    totp_secret=angelone_config.get("totp_secret", ""),
                )
                
                to_date = date.today()
                from_date = to_date - timedelta(days=days)
                
                candles = broker.get_historical(
                    symbol=symbol,
                    from_date=from_date,
                    to_date=to_date,
                    interval=timeframe,
                )
                
                if candles:
                    store.save_candles(symbol, timeframe, candles)
                    console.print(f"[dim]Fetched {len(candles)} candles[/dim]")
                    # Retry analysis
                    results = calculate_indicators_for_symbol(symbol, indicator_list, store, days=days, timeframe=timeframe)
        except Exception as e:
            console.print(f"[dim]Auto-fetch failed: {e}[/dim]")
    
    if "error" in results:
        console.print(Panel(
            f"[yellow]{results['error']}[/yellow]\n\n"
            "[dim]Try fetching data first with:[/dim]\n"
            f"[cyan]daytrader data {symbol} --days 5 -t {timeframe}[/cyan]",
            title="[bold yellow]Insufficient Data[/bold yellow]",
            border_style="yellow",
        ))
        return
    
    # Build output
    tf_label = tf_display[timeframe]
    output_lines = [
        f"[bold]{symbol}[/bold] - ₹{results['last_price']:.2f}",
        f"[dim]Based on {results['candle_count']} {tf_label} candles[/dim]\n",
    ]
    
    # Display each indicator
    ind_results = results.get("indicators", {})
    
    if "rsi" in ind_results:
        rsi = ind_results["rsi"]
        output_lines.append(
            f"[bold]RSI (14):[/bold] {rsi['value']:.2f} "
            f"[{rsi['color']}]→ {rsi['signal']}[/{rsi['color']}]"
        )
    
    if "macd" in ind_results:
        macd = ind_results["macd"]
        output_lines.append(
            f"[bold]MACD:[/bold] {macd['macd']:.4f} | Signal: {macd['signal']:.4f} | "
            f"Hist: {macd['histogram']:.4f}"
        )
        output_lines.append(
            f"       [{macd['color']}]→ {macd['interpretation']}[/{macd['color']}]"
        )
    
    if "ema" in ind_results:
        ema = ind_results["ema"]
        output_lines.append(
            f"[bold]EMA:[/bold] 9-day: ₹{ema['ema_9']:.2f} | 21-day: ₹{ema['ema_21']:.2f}"
        )
        output_lines.append(
            f"      [{ema['color']}]→ {ema['trend']} Trend[/{ema['color']}]"
        )
    
    if "sma" in ind_results:
        sma = ind_results["sma"]
        sma_50_str = f"₹{sma['sma_50']:.2f}" if sma['sma_50'] else "N/A"
        output_lines.append(
            f"[bold]SMA:[/bold] 20-day: ₹{sma['sma_20']:.2f} | 50-day: {sma_50_str}"
        )
        output_lines.append(
            f"      [{sma['color']}]→ {sma['trend']} (Price vs SMA20)[/{sma['color']}]"
        )
    
    if "bb" in ind_results:
        bb = ind_results["bb"]
        output_lines.append(
            f"[bold]Bollinger Bands:[/bold] Upper: ₹{bb['upper']:.2f} | "
            f"Middle: ₹{bb['middle']:.2f} | Lower: ₹{bb['lower']:.2f}"
        )
        output_lines.append(
            f"                 [{bb['color']}]→ {bb['signal']}[/{bb['color']}]"
        )
    
    if "atr" in ind_results:
        atr = ind_results["atr"]
        output_lines.append(
            f"[bold]ATR (14):[/bold] ₹{atr['value']:.2f} ({atr['percent']:.2f}%)"
        )
        output_lines.append(
            f"         [{atr['color']}]→ {atr['volatility']} Volatility[/{atr['color']}]"
        )
    
    if "vwap" in ind_results:
        vwap = ind_results["vwap"]
        output_lines.append(
            f"[bold]VWAP:[/bold] ₹{vwap['value']:.2f}"
        )
        output_lines.append(
            f"      [{vwap['color']}]→ {vwap['position']}[/{vwap['color']}]"
        )
    
    if "stoch" in ind_results:
        stoch = ind_results["stoch"]
        output_lines.append(
            f"[bold]Stochastic:[/bold] %K: {stoch['k']:.2f} | %D: {stoch['d']:.2f}"
        )
        output_lines.append(
            f"            [{stoch['color']}]→ {stoch['signal']}[/{stoch['color']}]"
        )
    
    if "adx" in ind_results:
        adx = ind_results["adx"]
        output_lines.append(
            f"[bold]ADX:[/bold] {adx['adx']:.2f} | +DI: {adx['plus_di']:.2f} | -DI: {adx['minus_di']:.2f}"
        )
        output_lines.append(
            f"     [{adx['color']}]→ {adx['trend_strength']} ({adx['direction']})[/{adx['color']}]"
        )
    
    if "supertrend" in ind_results:
        st = ind_results["supertrend"]
        output_lines.append(
            f"[bold]SuperTrend:[/bold] ₹{st['value']:.2f}"
        )
        output_lines.append(
            f"            [{st['color']}]→ {st['signal']}[/{st['color']}]"
        )
    
    if "fib" in ind_results:
        fib = ind_results["fib"]
        output_lines.append(
            f"[bold]Fibonacci ({fib['trend'].upper()} trend):[/bold] "
            f"Swing: ₹{fib['swing_low']:.2f} - ₹{fib['swing_high']:.2f}"
        )
        # Show key levels with current price position
        price = fib.get("price", results["last_price"])
        for level, level_price in fib["levels"].items():
            # Highlight nearest support/resistance
            marker = ""
            if fib.get("nearest_support") and fib["nearest_support"][0] == level:
                marker = " [green]◄ SUPPORT[/green]"
            elif fib.get("nearest_resistance") and fib["nearest_resistance"][0] == level:
                marker = " [red]◄ RESISTANCE[/red]"
            output_lines.append(f"  {level}: ₹{level_price:.2f}{marker}")
        
        # Show actionable info
        if fib.get("current_zone"):
            output_lines.append(f"  [cyan]Price Zone: {fib['current_zone']}[/cyan]")
        if fib.get("nearest_support"):
            support_dist = ((price - fib["nearest_support"][1]) / price) * 100
            output_lines.append(f"  [dim]Support {support_dist:.1f}% below[/dim]")
        if fib.get("nearest_resistance"):
            resist_dist = ((fib["nearest_resistance"][1] - price) / price) * 100
            output_lines.append(f"  [dim]Resistance {resist_dist:.1f}% above[/dim]")
    
    if "pivot" in ind_results:
        pivot = ind_results["pivot"]
        levels = pivot["levels"]
        output_lines.append(
            f"[bold]Pivot Points:[/bold] P: ₹{levels['Pivot']:.2f}"
        )
        output_lines.append(
            f"  R1: ₹{levels['R1']:.2f} | R2: ₹{levels['R2']:.2f} | R3: ₹{levels['R3']:.2f}"
        )
        output_lines.append(
            f"  S1: ₹{levels['S1']:.2f} | S2: ₹{levels['S2']:.2f} | S3: ₹{levels['S3']:.2f}"
        )
        output_lines.append(
            f"              [{pivot['color']}]→ {pivot['position']}[/{pivot['color']}]"
        )
    
    if "obv" in ind_results:
        obv = ind_results["obv"]
        output_lines.append(
            f"[bold]OBV:[/bold] {obv['value']:,.0f}"
        )
        output_lines.append(
            f"     [{obv['color']}]→ {obv['trend']} Volume[/{obv['color']}]"
        )
    
    if "cci" in ind_results:
        cci = ind_results["cci"]
        output_lines.append(
            f"[bold]CCI (20):[/bold] {cci['value']:.2f}"
        )
        output_lines.append(
            f"         [{cci['color']}]→ {cci['signal']}[/{cci['color']}]"
        )
    
    if "willr" in ind_results:
        willr = ind_results["willr"]
        output_lines.append(
            f"[bold]Williams %R:[/bold] {willr['value']:.2f}"
        )
        output_lines.append(
            f"            [{willr['color']}]→ {willr['signal']}[/{willr['color']}]"
        )
    
    if "patterns" in ind_results:
        patterns = ind_results["patterns"]
        if patterns["detected"]:
            output_lines.append(f"[bold]Candlestick Patterns:[/bold]")
            for p in patterns["detected"]:
                color = "green" if p["signal"] == "bullish" else "red" if p["signal"] == "bearish" else "yellow"
                output_lines.append(f"  [{color}]• {p['pattern']} ({p['signal']})[/{color}]")
        else:
            output_lines.append(f"[bold]Candlestick Patterns:[/bold] [dim]None detected[/dim]")
    
    # Add trading summary
    output_lines.append("")
    output_lines.append("[bold]─── Trading Summary ───[/bold]")
    
    bullish_signals = 0
    bearish_signals = 0
    sl_price = None
    target_price = None
    
    # Count signals and extract key levels
    if "supertrend" in ind_results:
        st = ind_results["supertrend"]
        if st["direction"] == 1:
            bullish_signals += 1
            sl_price = st["value"]  # SuperTrend as SL
        else:
            bearish_signals += 1
    
    if "rsi" in ind_results:
        rsi = ind_results["rsi"]
        if rsi["value"] < 40:
            bullish_signals += 1
        elif rsi["value"] > 60:
            bearish_signals += 1
    
    if "adx" in ind_results:
        adx = ind_results["adx"]
        if adx["direction"] == "Bullish" and adx["adx"] > 20:
            bullish_signals += 1
        elif adx["direction"] == "Bearish" and adx["adx"] > 20:
            bearish_signals += 1
    
    if "bb" in ind_results:
        bb = ind_results["bb"]
        if "Lower" in bb["signal"]:
            bullish_signals += 1
        elif "Upper" in bb["signal"]:
            bearish_signals += 1
    
    # Get target from Fibonacci resistance
    if "fib" in ind_results:
        fib = ind_results["fib"]
        if fib.get("nearest_resistance"):
            target_price = fib["nearest_resistance"][1]
        if fib.get("nearest_support") and not sl_price:
            sl_price = fib["nearest_support"][1]
    
    # Determine overall bias
    total_signals = bullish_signals + bearish_signals
    if total_signals > 0:
        if bullish_signals > bearish_signals:
            bias = "BULLISH"
            bias_color = "green"
            bias_pct = (bullish_signals / total_signals) * 100
        elif bearish_signals > bullish_signals:
            bias = "BEARISH"
            bias_color = "red"
            bias_pct = (bearish_signals / total_signals) * 100
        else:
            bias = "NEUTRAL"
            bias_color = "yellow"
            bias_pct = 50
        
        output_lines.append(f"Bias: [{bias_color}]{bias}[/{bias_color}] ({bias_pct:.0f}% of signals)")
    
    # Show suggested levels
    price = results["last_price"]
    if sl_price:
        sl_risk = ((price - sl_price) / price) * 100
        output_lines.append(f"Stop Loss: ₹{sl_price:.2f} [dim]({sl_risk:.1f}% risk)[/dim]")
    if target_price and target_price > price:
        target_reward = ((target_price - price) / price) * 100
        output_lines.append(f"Target: ₹{target_price:.2f} [dim]({target_reward:.1f}% reward)[/dim]")
        if sl_price and sl_price < price:
            rr_ratio = target_reward / sl_risk if sl_risk > 0 else 0
            rr_color = "green" if rr_ratio >= 2 else "yellow" if rr_ratio >= 1 else "red"
            output_lines.append(f"Risk:Reward: [{rr_color}]1:{rr_ratio:.1f}[/{rr_color}]")
    
    console.print(Panel(
        "\n".join(output_lines),
        title="[bold cyan]Technical Analysis[/bold cyan]",
        border_style="cyan",
    ))
    
    # AI verification if requested
    if ai:
        console.print("\n[dim]Getting AI analysis...[/dim]")
        ai_response = _get_ai_analysis(symbol, timeframe, results, config)
        console.print(Panel(
            ai_response,
            title="[bold magenta]🤖 AI Analysis[/bold magenta]",
            border_style="magenta",
        ))


@click.command()
@click.argument("symbol")
def signal(symbol: str) -> None:
    """Get a composite buy/sell signal score for a symbol.
    
    Combines multiple indicators (RSI, MACD, Stochastic, SuperTrend, ADX, MAs)
    into a single score from -100 (strong sell) to +100 (strong buy).
    
    SYMBOL is the trading symbol (e.g., RELIANCE, INFY, TCS).
    
    \b
    Score interpretation:
      +50 to +100  : STRONG BUY
      +25 to +50   : BUY
      +10 to +25   : WEAK BUY
      -10 to +10   : NEUTRAL
      -25 to -10   : WEAK SELL
      -50 to -25   : SELL
      -100 to -50  : STRONG SELL
    
    \b
    Examples:
      daytrader signal RELIANCE
      daytrader signal INFY
    """
    config = _get_config()
    
    if config is None:
        console.print(Panel(
            "[red]Configuration not found.[/red]\n\n"
            "Run [cyan]daytrader login[/cyan] to create a config file.",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    symbol = symbol.upper()
    store = _get_data_store()
    
    console.print(f"[dim]Calculating signal score for {symbol}...[/dim]")
    
    # Get candle data
    to_date = date.today()
    from_date = to_date - timedelta(days=100)
    candles = store.get_candles(symbol, "1day", from_date, to_date)
    
    if not candles or len(candles) < 50:
        console.print(Panel(
            f"[yellow]Insufficient data for {symbol}. Need at least 50 candles.[/yellow]\n\n"
            "[dim]Try fetching data first with:[/dim]\n"
            f"[cyan]daytrader data {symbol} --days 100[/cyan]",
            title="[bold yellow]Insufficient Data[/bold yellow]",
            border_style="yellow",
        ))
        return
    
    # Extract price arrays
    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [float(c.volume) for c in candles]
    
    # Calculate signal score
    result = calculate_signal_score(highs, lows, closes, volumes)
    
    if "error" in result:
        console.print(Panel(
            f"[red]{result['error']}[/red]",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        return
    
    # Build score bar visualization
    score = result["total_score"]
    bar_width = 40
    bar_position = int((score + 100) / 200 * bar_width)
    bar_position = max(0, min(bar_width, bar_position))
    
    bar = ""
    for i in range(bar_width):
        if i < bar_width // 2:
            if i < bar_position:
                bar += "[red]█[/red]"
            else:
                bar += "[dim]░[/dim]"
        else:
            if i < bar_position:
                bar += "[green]█[/green]"
            else:
                bar += "[dim]░[/dim]"
    
    # Get market status
    market_status, market_color, is_market_open = _get_market_status()
    
    # Build output
    output_lines = [
        f"[bold]{symbol}[/bold] - ₹{closes[-1]:.2f}",
        f"[bold]Market:[/bold] [{market_color}]{market_status}[/{market_color}]\n",
        f"[bold]Signal Score:[/bold] [{result['color']}]{score:+.1f}[/{result['color']}]",
        f"[bold]Recommendation:[/bold] [{result['color']}]{result['recommendation']}[/{result['color']}]",
    ]
    
    # Add warning if market is closed
    if not is_market_open:
        output_lines.append(f"[yellow]⚠ Market is closed - signals based on last trading session[/yellow]")
    
    output_lines.extend([
        "",
        f"SELL {bar} BUY",
        f"-100 {'─' * 16} 0 {'─' * 16} +100\n",
        "[bold]Indicator Breakdown:[/bold]",
    ])
    
    # Show breakdown
    breakdown = result["breakdown"]
    for indicator, score_val in breakdown.items():
        indicator_name = {
            "rsi": "RSI",
            "macd": "MACD",
            "stoch": "Stochastic",
            "supertrend": "SuperTrend",
            "adx": "ADX",
            "ma": "Moving Avg",
            "volume": "Volume",
        }.get(indicator, indicator.upper())
        
        color = "green" if score_val > 0 else "red" if score_val < 0 else "dim"
        output_lines.append(f"  {indicator_name:12} [{color}]{score_val:+.0f}[/{color}]")
    
    console.print(Panel(
        "\n".join(output_lines),
        title="[bold cyan]Signal Analysis[/bold cyan]",
        border_style="cyan",
    ))


@click.command()
@click.argument("symbol")
def mtf(symbol: str) -> None:
    """Multi-timeframe analysis showing signals across different timeframes.
    
    Shows RSI, MACD, and SuperTrend signals for 5min, 15min, 1hour, and 1day
    timeframes to help identify trend alignment.
    
    SYMBOL is the trading symbol (e.g., RELIANCE, INFY, TCS).
    
    \b
    Examples:
      daytrader mtf RELIANCE
      daytrader mtf INFY
    """
    config = _get_config()
    
    if config is None:
        console.print(Panel(
            "[red]Configuration not found.[/red]\n\n"
            "Run [cyan]daytrader login[/cyan] to create a config file.",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    symbol = symbol.upper()
    store = _get_data_store()
    
    # Get market status
    market_status, market_color, is_market_open = _get_market_status()
    
    console.print(f"[dim]Fetching multi-timeframe data for {symbol}...[/dim]")
    console.print(f"[bold]Market:[/bold] [{market_color}]{market_status}[/{market_color}]\n")
    
    if not is_market_open:
        console.print("[yellow]⚠ Market is closed - intraday signals may be stale[/yellow]\n")
    
    # Try to get data broker for fetching
    from daytrader.brokers.angelone import AngelOneBroker
    angelone_config = config.get("angelone", {})
    
    if not angelone_config.get("api_key"):
        console.print(Panel(
            "[yellow]Angel One not configured. Multi-timeframe requires live data.[/yellow]",
            title="[bold yellow]Configuration Required[/bold yellow]",
            border_style="yellow",
        ))
        return
    
    broker = AngelOneBroker(
        api_key=angelone_config.get("api_key", ""),
        client_id=angelone_config.get("client_id", ""),
        pin=angelone_config.get("pin", ""),
        totp_secret=angelone_config.get("totp_secret", ""),
    )
    
    timeframes = [
        ("5min", 5),
        ("15min", 3),
        ("1hour", 7),
        ("1day", 100),
    ]
    
    # Create table
    table = Table(
        title=f"Multi-Timeframe Analysis: {symbol}",
        show_header=True,
        header_style="bold cyan",
    )
    
    table.add_column("Timeframe", style="bold")
    table.add_column("RSI", justify="center")
    table.add_column("MACD", justify="center")
    table.add_column("SuperTrend", justify="center")
    table.add_column("Signal", justify="center")
    
    alignment_scores = []
    
    for tf_name, days in timeframes:
        try:
            # Get data
            to_date = date.today()
            from_date = to_date - timedelta(days=days)
            
            # Try cache first, then fetch
            candles = store.get_candles(symbol, tf_name, from_date, to_date)
            
            if not candles or len(candles) < 30:
                candles = broker.get_historical(symbol, from_date, to_date, tf_name)
                if candles:
                    store.save_candles(symbol, tf_name, candles)
            
            if not candles or len(candles) < 20:
                table.add_row(tf_name, "[dim]N/A[/dim]", "[dim]N/A[/dim]", "[dim]N/A[/dim]", "[dim]N/A[/dim]")
                continue
            
            closes = [c.close for c in candles]
            highs = [c.high for c in candles]
            lows = [c.low for c in candles]
            
            # Calculate indicators
            rsi_values = calculate_rsi(closes, 14)
            macd_line, signal_line, histogram = calculate_macd(closes)
            st_values, st_direction = calculate_supertrend(highs, lows, closes)
            
            # RSI signal
            rsi = rsi_values[-1] if rsi_values and rsi_values[-1] == rsi_values[-1] else None
            if rsi:
                if rsi < 30:
                    rsi_signal = "[green]Oversold[/green]"
                    rsi_score = 1
                elif rsi > 70:
                    rsi_signal = "[red]Overbought[/red]"
                    rsi_score = -1
                elif rsi < 50:
                    rsi_signal = "[yellow]Bearish[/yellow]"
                    rsi_score = -0.5
                else:
                    rsi_signal = "[yellow]Bullish[/yellow]"
                    rsi_score = 0.5
            else:
                rsi_signal = "[dim]N/A[/dim]"
                rsi_score = 0
            
            # MACD signal
            hist = histogram[-1] if histogram and histogram[-1] == histogram[-1] else None
            if hist is not None:
                if hist > 0:
                    macd_signal = "[green]Bullish[/green]"
                    macd_score = 1
                else:
                    macd_signal = "[red]Bearish[/red]"
                    macd_score = -1
            else:
                macd_signal = "[dim]N/A[/dim]"
                macd_score = 0
            
            # SuperTrend signal
            st_dir = st_direction[-1] if st_direction and st_direction[-1] != 0 else None
            if st_dir:
                if st_dir == 1:
                    st_signal = "[green]BUY[/green]"
                    st_score = 1
                else:
                    st_signal = "[red]SELL[/red]"
                    st_score = -1
            else:
                st_signal = "[dim]N/A[/dim]"
                st_score = 0
            
            # Combined signal for this timeframe
            tf_score = rsi_score + macd_score + st_score
            alignment_scores.append(tf_score)
            
            if tf_score >= 2:
                combined = "[green]▲ BUY[/green]"
            elif tf_score <= -2:
                combined = "[red]▼ SELL[/red]"
            elif tf_score > 0:
                combined = "[yellow]↗ Weak Buy[/yellow]"
            elif tf_score < 0:
                combined = "[yellow]↘ Weak Sell[/yellow]"
            else:
                combined = "[dim]— Neutral[/dim]"
            
            table.add_row(tf_name, rsi_signal, macd_signal, st_signal, combined)
            
        except Exception as e:
            table.add_row(tf_name, "[red]Error[/red]", "[red]Error[/red]", "[red]Error[/red]", f"[dim]{str(e)[:20]}[/dim]")
    
    console.print(table)
    
    # Overall alignment
    if alignment_scores:
        avg_score = sum(alignment_scores) / len(alignment_scores)
        
        if avg_score >= 1.5:
            alignment = "[green]STRONG BULLISH ALIGNMENT[/green] - All timeframes agree on BUY"
        elif avg_score >= 0.5:
            alignment = "[green]BULLISH BIAS[/green] - Most timeframes favor buying"
        elif avg_score <= -1.5:
            alignment = "[red]STRONG BEARISH ALIGNMENT[/red] - All timeframes agree on SELL"
        elif avg_score <= -0.5:
            alignment = "[red]BEARISH BIAS[/red] - Most timeframes favor selling"
        else:
            alignment = "[yellow]MIXED SIGNALS[/yellow] - Timeframes disagree, wait for clarity"
        
        console.print(f"\n[bold]Overall:[/bold] {alignment}")
