"""Scan command for DayTrader CLI.

Scans watchlist stocks based on technical criteria to find trading opportunities.
"""

from datetime import date, timedelta
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from daytrader.indicators.technical import calculate_rsi

console = Console()


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


def get_stock_metrics(symbol: str, store) -> Optional[dict]:
    """Get metrics for a stock from cached data.
    
    Args:
        symbol: Trading symbol.
        store: DataStore instance.
        
    Returns:
        Dictionary with metrics or None if insufficient data.
    """
    # Get candle data from cache
    to_date = date.today()
    from_date = to_date - timedelta(days=100)
    
    candles = store.get_candles(symbol, "1day", from_date, to_date)
    
    if not candles or len(candles) < 20:
        return None
    
    # Extract price arrays
    closes = [c.close for c in candles]
    volumes = [c.volume for c in candles]
    
    # Calculate metrics
    current_price = closes[-1]
    prev_close = closes[-2] if len(closes) > 1 else closes[-1]
    
    # Gap calculation (today's open vs yesterday's close)
    today_open = candles[-1].open
    gap_percent = ((today_open - prev_close) / prev_close * 100) if prev_close > 0 else 0
    
    # RSI
    rsi_values = calculate_rsi(closes, period=14)
    rsi = rsi_values[-1] if rsi_values and rsi_values[-1] == rsi_values[-1] else None
    
    # Volume analysis
    avg_volume = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else volumes[0]
    current_volume = volumes[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
    
    # Price change
    change = current_price - prev_close
    change_percent = (change / prev_close * 100) if prev_close > 0 else 0
    
    return {
        "symbol": symbol,
        "price": current_price,
        "change": change,
        "change_percent": change_percent,
        "gap_percent": gap_percent,
        "rsi": rsi,
        "volume": current_volume,
        "avg_volume": avg_volume,
        "volume_ratio": volume_ratio,
    }


def scan_stocks(
    symbols: list[str],
    store,
    rsi_below: Optional[float] = None,
    rsi_above: Optional[float] = None,
    gap_up: Optional[float] = None,
    gap_down: Optional[float] = None,
    volume_spike: bool = False,
) -> list[dict]:
    """Scan stocks based on filter criteria.
    
    Args:
        symbols: List of symbols to scan.
        store: DataStore instance.
        rsi_below: Filter for RSI below this value.
        rsi_above: Filter for RSI above this value.
        gap_up: Filter for gap up greater than this percentage.
        gap_down: Filter for gap down greater than this percentage.
        volume_spike: Filter for volume above 2x average.
        
    Returns:
        List of stocks matching the criteria with their metrics.
    """
    results = []
    
    for symbol in symbols:
        metrics = get_stock_metrics(symbol, store)
        
        if metrics is None:
            continue
        
        # Apply filters
        passes_filter = True
        match_reasons = []
        
        if rsi_below is not None:
            if metrics["rsi"] is None or metrics["rsi"] >= rsi_below:
                passes_filter = False
            else:
                match_reasons.append(f"RSI {metrics['rsi']:.1f} < {rsi_below}")
        
        if rsi_above is not None:
            if metrics["rsi"] is None or metrics["rsi"] <= rsi_above:
                passes_filter = False
            else:
                match_reasons.append(f"RSI {metrics['rsi']:.1f} > {rsi_above}")
        
        if gap_up is not None:
            if metrics["gap_percent"] <= gap_up:
                passes_filter = False
            else:
                match_reasons.append(f"Gap Up {metrics['gap_percent']:.2f}%")
        
        if gap_down is not None:
            if metrics["gap_percent"] >= -gap_down:
                passes_filter = False
            else:
                match_reasons.append(f"Gap Down {abs(metrics['gap_percent']):.2f}%")
        
        if volume_spike:
            if metrics["volume_ratio"] < 2.0:
                passes_filter = False
            else:
                match_reasons.append(f"Volume {metrics['volume_ratio']:.1f}x avg")
        
        if passes_filter:
            metrics["match_reasons"] = match_reasons
            results.append(metrics)
    
    # Sort by relevance (number of match reasons, then by RSI for oversold)
    results.sort(key=lambda x: (-len(x.get("match_reasons", [])), x.get("rsi", 50) or 50))
    
    return results


@click.command()
@click.option(
    "-rb", "--rsi-below",
    type=float,
    help="Filter stocks with RSI below this value (e.g., 30 for oversold)",
)
@click.option(
    "-ra", "--rsi-above",
    type=float,
    help="Filter stocks with RSI above this value (e.g., 70 for overbought)",
)
@click.option(
    "-gu", "--gap-up",
    type=float,
    help="Filter stocks gapping up more than this percentage",
)
@click.option(
    "-gd", "--gap-down",
    type=float,
    help="Filter stocks gapping down more than this percentage",
)
@click.option(
    "-v", "--volume-spike",
    is_flag=True,
    help="Filter stocks with volume above 2x average",
)
@click.option(
    "-w", "--watchlist",
    default="default",
    help="Watchlist to scan (default: 'default')",
)
def scan(
    rsi_below: Optional[float],
    rsi_above: Optional[float],
    gap_up: Optional[float],
    gap_down: Optional[float],
    volume_spike: bool,
    watchlist: str,
) -> None:
    """Scan watchlist stocks based on technical criteria.
    
    Scans stocks in your watchlist and filters based on technical
    indicators like RSI, gap percentage, and volume.
    
    \b
    Examples:
      daytrader scan --rsi-below 30        # Find oversold stocks
      daytrader scan --rsi-above 70        # Find overbought stocks
      daytrader scan --gap-up 2            # Stocks gapping up > 2%
      daytrader scan --volume-spike        # Volume > 2x average
      daytrader scan --rsi-below 30 --volume-spike  # Combined filters
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
    
    # Check if any filter is specified
    if not any([rsi_below, rsi_above, gap_up, gap_down, volume_spike]):
        console.print(Panel(
            "[yellow]No filter specified.[/yellow]\n\n"
            "Please specify at least one filter:\n"
            "  [cyan]--rsi-below 30[/cyan]    Oversold stocks\n"
            "  [cyan]--rsi-above 70[/cyan]    Overbought stocks\n"
            "  [cyan]--gap-up 2[/cyan]        Gap up > 2%\n"
            "  [cyan]--gap-down 2[/cyan]      Gap down > 2%\n"
            "  [cyan]--volume-spike[/cyan]    Volume > 2x average",
            title="[bold yellow]Filter Required[/bold yellow]",
            border_style="yellow",
        ))
        return
    
    store = _get_data_store()
    
    # Get watchlist symbols
    symbols = store.get_watchlist(watchlist)
    
    if not symbols:
        console.print(Panel(
            f"[yellow]Watchlist '{watchlist}' is empty.[/yellow]\n\n"
            "Add stocks to your watchlist first:\n"
            "[cyan]daytrader watch add RELIANCE[/cyan]\n"
            "[cyan]daytrader watch add INFY[/cyan]",
            title="[bold yellow]Empty Watchlist[/bold yellow]",
            border_style="yellow",
        ))
        return
    
    console.print(f"[dim]Scanning {len(symbols)} stocks in '{watchlist}' watchlist...[/dim]")
    
    # Perform scan
    results = scan_stocks(
        symbols=symbols,
        store=store,
        rsi_below=rsi_below,
        rsi_above=rsi_above,
        gap_up=gap_up,
        gap_down=gap_down,
        volume_spike=volume_spike,
    )
    
    if not results:
        console.print(Panel(
            "[dim]No stocks match the specified criteria.[/dim]\n\n"
            "Try adjusting your filters or ensure you have\n"
            "recent data for your watchlist stocks.",
            title="[bold]No Results[/bold]",
            border_style="dim",
        ))
        return
    
    # Display results
    table = Table(
        title=f"Scan Results ({len(results)} matches)",
        show_header=True,
        header_style="bold cyan",
    )
    
    table.add_column("Symbol", style="bold")
    table.add_column("Price", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("RSI", justify="right")
    table.add_column("Gap", justify="right")
    table.add_column("Vol Ratio", justify="right")
    table.add_column("Match Reasons", style="dim")
    
    for stock in results:
        # Format change
        change_str = f"{stock['change']:+.2f} ({stock['change_percent']:+.2f}%)"
        change_style = "green" if stock['change'] >= 0 else "red"
        
        # Format RSI
        rsi_str = f"{stock['rsi']:.1f}" if stock['rsi'] else "N/A"
        if stock['rsi']:
            if stock['rsi'] < 30:
                rsi_style = "green"
            elif stock['rsi'] > 70:
                rsi_style = "red"
            else:
                rsi_style = "dim"
        else:
            rsi_style = "dim"
        
        # Format gap
        gap_str = f"{stock['gap_percent']:+.2f}%"
        gap_style = "green" if stock['gap_percent'] > 0 else "red" if stock['gap_percent'] < 0 else "dim"
        
        # Format volume ratio
        vol_str = f"{stock['volume_ratio']:.1f}x"
        vol_style = "yellow" if stock['volume_ratio'] >= 2.0 else "dim"
        
        # Match reasons
        reasons = ", ".join(stock.get("match_reasons", []))
        
        table.add_row(
            stock["symbol"],
            f"₹{stock['price']:.2f}",
            f"[{change_style}]{change_str}[/{change_style}]",
            f"[{rsi_style}]{rsi_str}[/{rsi_style}]",
            f"[{gap_style}]{gap_str}[/{gap_style}]",
            f"[{vol_style}]{vol_str}[/{vol_style}]",
            reasons,
        )
    
    console.print(table)
