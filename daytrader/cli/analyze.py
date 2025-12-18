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
)

console = Console()

# Available indicators
AVAILABLE_INDICATORS = ["rsi", "macd", "ema", "sma", "bb", "atr", "vwap"]
DEFAULT_INDICATORS = ["rsi", "macd", "bb"]


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
) -> dict:
    """Calculate requested indicators for a symbol.
    
    Args:
        symbol: Trading symbol.
        indicators: List of indicator names to calculate.
        store: DataStore instance.
        days: Number of days of data to use.
        
    Returns:
        Dictionary with indicator results.
    """
    # Get candle data from cache
    to_date = date.today()
    from_date = to_date - timedelta(days=days)
    
    candles = store.get_candles(symbol, "1day", from_date, to_date)
    
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
    
    return results


@click.command()
@click.argument("symbol")
@click.option(
    "--indicators",
    "-i",
    default=",".join(DEFAULT_INDICATORS),
    help=f"Comma-separated indicators to calculate. Available: {', '.join(AVAILABLE_INDICATORS)} (default: {', '.join(DEFAULT_INDICATORS)})",
)
def analyze(symbol: str, indicators: str) -> None:
    """Calculate and display technical indicators for a symbol.
    
    SYMBOL is the trading symbol (e.g., RELIANCE, INFY, TCS).
    
    \b
    Available indicators:
      rsi   - Relative Strength Index (14-period)
      macd  - Moving Average Convergence Divergence
      ema   - Exponential Moving Averages (9, 21)
      sma   - Simple Moving Averages (20, 50)
      bb    - Bollinger Bands (20-period, 2 std dev)
      atr   - Average True Range (14-period)
      vwap  - Volume Weighted Average Price
    
    \b
    Examples:
      daytrader analyze RELIANCE                    # Default indicators
      daytrader analyze INFY --indicators rsi,macd # Specific indicators
      daytrader analyze TCS -i rsi,bb,atr          # Short form
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
    indicator_list = _parse_indicators(indicators)
    
    console.print(f"[dim]Analyzing {symbol} with indicators: {', '.join(indicator_list)}...[/dim]")
    
    store = _get_data_store()
    results = calculate_indicators_for_symbol(symbol, indicator_list, store)
    
    if "error" in results:
        console.print(Panel(
            f"[yellow]{results['error']}[/yellow]\n\n"
            "[dim]Try fetching data first with:[/dim]\n"
            f"[cyan]daytrader data {symbol} --days 100[/cyan]",
            title="[bold yellow]Insufficient Data[/bold yellow]",
            border_style="yellow",
        ))
        return
    
    # Build output
    output_lines = [
        f"[bold]{symbol}[/bold] - ₹{results['last_price']:.2f}",
        f"[dim]Based on {results['candle_count']} daily candles[/dim]\n",
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
    
    console.print(Panel(
        "\n".join(output_lines),
        title="[bold cyan]Technical Analysis[/bold cyan]",
        border_style="cyan",
    ))
