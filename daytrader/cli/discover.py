"""Discover command for DayTrader CLI.

Shows market movers like top gainers, losers, and most active stocks.
"""

from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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


def _get_broker(config: dict):
    """Get the Angel One broker for market data."""
    from daytrader.brokers.angelone import AngelOneBroker
    
    angelone_config = config.get("angelone", {})
    return AngelOneBroker(
        api_key=angelone_config.get("api_key", ""),
        client_id=angelone_config.get("client_id", ""),
        pin=angelone_config.get("pin", ""),
        totp_secret=angelone_config.get("totp_secret", ""),
    )


# Popular NSE stocks for scanning (Nifty 50 + some popular mid-caps)
SCAN_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK", "BAJFINANCE", "MARUTI",
    "ASIANPAINT", "TITAN", "WIPRO", "HCLTECH", "SUNPHARMA", "TATAMOTORS",
    "TATASTEEL", "POWERGRID", "NTPC", "ONGC", "COALINDIA", "BAJAJFINSV",
    "ADANIENT", "ADANIPORTS", "ULTRACEMCO", "NESTLEIND", "JSWSTEEL", "TECHM",
    "GRASIM", "INDUSINDBK", "HINDALCO", "DRREDDY", "CIPLA", "EICHERMOT",
    "DIVISLAB", "BPCL", "BRITANNIA", "APOLLOHOSP", "HEROMOTOCO", "TATACONSUM",
    "SBILIFE", "BAJAJ-AUTO", "YESBANK", "IDEA", "ZOMATO", "PAYTM"
]


@click.command()
@click.option(
    "-g", "--gainers",
    is_flag=True,
    default=False,
    help="Show top gainers only.",
)
@click.option(
    "-l", "--losers",
    is_flag=True,
    default=False,
    help="Show top losers only.",
)
@click.option(
    "-a", "--active",
    is_flag=True,
    default=False,
    help="Show most active by volume.",
)
@click.option(
    "-n", "--limit",
    type=int,
    default=10,
    help="Number of stocks to show (default: 10).",
)
def discover(gainers: bool, losers: bool, active: bool, limit: int) -> None:
    """Discover market movers - top gainers, losers, and active stocks.
    
    Scans popular NSE stocks and shows market movers based on today's
    price action.
    
    \b
    Examples:
      daytrader discover              # Show all categories
      daytrader discover --gainers    # Top gainers only
      daytrader discover --losers     # Top losers only
      daytrader discover --active     # Most active by volume
      daytrader discover --limit 20   # Show top 20
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
    
    # If no specific filter, show all
    show_all = not (gainers or losers or active)
    
    console.print("[dim]Scanning market for movers...[/dim]")
    
    try:
        broker = _get_broker(config)
        
        # Fetch quotes for all stocks in universe
        quotes = []
        failed = 0
        
        with console.status("[bold green]Fetching quotes...") as status:
            for i, symbol in enumerate(SCAN_UNIVERSE):
                status.update(f"[bold green]Fetching {symbol} ({i+1}/{len(SCAN_UNIVERSE)})...")
                try:
                    q = broker.get_quote(symbol)
                    quotes.append({
                        "symbol": symbol,
                        "ltp": q.ltp,
                        "change": q.change,
                        "change_pct": q.change_percent,
                        "volume": q.volume,
                        "open": q.open,
                        "high": q.high,
                        "low": q.low,
                    })
                except Exception:
                    failed += 1
                    continue
        
        if not quotes:
            console.print(Panel(
                "[yellow]Could not fetch any quotes.[/yellow]\n\n"
                "[dim]Make sure you're logged in and market data is available.[/dim]",
                title="[bold yellow]No Data[/bold yellow]",
                border_style="yellow",
            ))
            return
        
        console.print(f"[dim]Fetched {len(quotes)} quotes ({failed} failed)[/dim]\n")
        
        # Sort for different views
        gainers_list = sorted(quotes, key=lambda x: x["change_pct"], reverse=True)
        losers_list = sorted(quotes, key=lambda x: x["change_pct"])
        active_list = sorted(quotes, key=lambda x: x["volume"], reverse=True)
        
        # Display results
        if show_all or gainers:
            _display_table(
                "🚀 Top Gainers",
                gainers_list[:limit],
                "green",
            )
        
        if show_all or losers:
            _display_table(
                "📉 Top Losers",
                losers_list[:limit],
                "red",
            )
        
        if show_all or active:
            _display_table(
                "🔥 Most Active (by Volume)",
                active_list[:limit],
                "yellow",
                show_volume=True,
            )
        
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to discover stocks:[/red]\n\n{str(e)}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)


def _display_table(title: str, stocks: list, color: str, show_volume: bool = False) -> None:
    """Display a table of stocks.
    
    Args:
        title: Table title.
        stocks: List of stock data dicts.
        color: Border color.
        show_volume: Whether to highlight volume column.
    """
    table = Table(
        title=title,
        show_header=True,
        header_style=f"bold {color}",
        border_style=color,
    )
    
    table.add_column("#", style="dim", width=3)
    table.add_column("Symbol", style="bold")
    table.add_column("LTP", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("Change %", justify="right")
    table.add_column("Open", justify="right", style="dim")
    table.add_column("High", justify="right", style="green")
    table.add_column("Low", justify="right", style="red")
    if show_volume:
        table.add_column("Volume", justify="right", style="yellow")
    
    for i, stock in enumerate(stocks, 1):
        change_color = "green" if stock["change"] >= 0 else "red"
        change_sign = "+" if stock["change"] >= 0 else ""
        
        row = [
            str(i),
            stock["symbol"],
            f"₹{stock['ltp']:.2f}",
            f"[{change_color}]{change_sign}{stock['change']:.2f}[/{change_color}]",
            f"[{change_color}]{change_sign}{stock['change_pct']:.2f}%[/{change_color}]",
            f"₹{stock['open']:.2f}",
            f"₹{stock['high']:.2f}",
            f"₹{stock['low']:.2f}",
        ]
        
        if show_volume:
            row.append(f"{stock['volume']:,}")
        
        table.add_row(*row)
    
    console.print(table)
    console.print()
