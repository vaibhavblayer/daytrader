"""Data commands for DayTrader CLI.

Handles fetching and displaying market data including historical OHLCV
data and real-time quotes.
"""

from datetime import date, timedelta

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# Supported timeframes
VALID_TIMEFRAMES = ["1min", "5min", "15min", "1hour", "1day"]


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


def _get_broker(config: dict):
    """Get the appropriate broker based on config."""
    trading_mode = config.get("trading", {}).get("mode", "paper")
    
    if trading_mode == "paper":
        from daytrader.brokers.paper import PaperBroker
        
        store = _get_data_store()
        starting_balance = config.get("trading", {}).get("paper_starting_balance", 100000.0)
        return PaperBroker(store, starting_balance=starting_balance)
    else:
        from daytrader.brokers.angelone import AngelOneBroker
        
        angelone_config = config.get("angelone", {})
        return AngelOneBroker(
            api_key=angelone_config.get("api_key", ""),
            client_id=angelone_config.get("client_id", ""),
            pin=angelone_config.get("pin", ""),
            totp_secret=angelone_config.get("totp_secret", ""),
        )


def _get_data_broker(config: dict):
    """Get Angel One broker for data fetching (even in paper mode).
    
    This allows fetching real market data while still using paper trading.
    """
    from daytrader.brokers.angelone import AngelOneBroker
    
    angelone_config = config.get("angelone", {})
    
    # Check if Angel One credentials are configured
    if not angelone_config.get("api_key"):
        return None
    
    return AngelOneBroker(
        api_key=angelone_config.get("api_key", ""),
        client_id=angelone_config.get("client_id", ""),
        pin=angelone_config.get("pin", ""),
        totp_secret=angelone_config.get("totp_secret", ""),
    )


@click.command()
@click.argument("symbol")
@click.option(
    "-d", "--days",
    default=30,
    type=int,
    help="Number of days of historical data to fetch (default: 30)",
)
@click.option(
    "-t", "--timeframe",
    default="1day",
    type=click.Choice(VALID_TIMEFRAMES),
    help="Candle timeframe (default: 1day)",
)
def data(symbol: str, days: int, timeframe: str) -> None:
    """Fetch and display historical OHLCV data for a symbol.
    
    SYMBOL is the trading symbol (e.g., RELIANCE, INFY, TCS).
    
    \b
    Examples:
      daytrader data RELIANCE              # Last 30 days, daily candles
      daytrader data INFY --days 7         # Last 7 days
      daytrader data TCS --timeframe 5min  # 5-minute candles
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
    
    # Calculate date range
    to_date = date.today()
    from_date = to_date - timedelta(days=days)
    
    console.print(f"[dim]Fetching {timeframe} data for {symbol}...[/dim]")
    
    # First check cache
    cached_candles = store.get_candles(
        symbol=symbol,
        timeframe=timeframe,
        from_date=from_date,
        to_date=to_date,
    )
    
    if cached_candles:
        console.print(f"[dim]Found {len(cached_candles)} candles in cache[/dim]")
        candles = cached_candles
    else:
        # Fetch from broker - use Angel One for data even in paper mode
        try:
            # Try data broker first (Angel One for real data)
            data_broker = _get_data_broker(config)
            if data_broker:
                candles = data_broker.get_historical(
                    symbol=symbol,
                    from_date=from_date,
                    to_date=to_date,
                    interval=timeframe,
                )
            else:
                # Fall back to regular broker
                broker = _get_broker(config)
                candles = broker.get_historical(
                    symbol=symbol,
                    from_date=from_date,
                    to_date=to_date,
                    interval=timeframe,
                )
            
            if candles:
                # Cache the data
                store.save_candles(symbol, timeframe, candles)
                console.print(f"[dim]Fetched and cached {len(candles)} candles[/dim]")
            else:
                console.print(Panel(
                    f"[yellow]No data available for {symbol}[/yellow]\n\n"
                    "[dim]The symbol may be invalid or no trading data exists "
                    "for the requested period.\n\n"
                    "Make sure Angel One credentials are configured in config.toml "
                    "to fetch real market data.[/dim]",
                    title="[bold yellow]No Data[/bold yellow]",
                    border_style="yellow",
                ))
                return
                
        except Exception as e:
            console.print(Panel(
                f"[red]Failed to fetch data:[/red]\n\n{str(e)}",
                title="[bold red]Error[/bold red]",
                border_style="red",
            ))
            raise SystemExit(1)
    
    # Display data in a table
    table = Table(
        title=f"{symbol} - {timeframe} ({len(candles)} candles)",
        show_header=True,
        header_style="bold cyan",
    )
    
    table.add_column("Date/Time", style="dim")
    table.add_column("Open", justify="right")
    table.add_column("High", justify="right", style="green")
    table.add_column("Low", justify="right", style="red")
    table.add_column("Close", justify="right")
    table.add_column("Volume", justify="right", style="dim")
    table.add_column("Change", justify="right")
    
    # Show last 20 candles by default
    display_candles = candles[-20:] if len(candles) > 20 else candles
    
    for i, candle in enumerate(display_candles):
        # Calculate change from previous candle
        if i > 0 or (len(candles) > 20 and len(display_candles) > 0):
            prev_idx = candles.index(candle) - 1 if candle in candles else i - 1
            if prev_idx >= 0 and prev_idx < len(candles):
                prev_close = candles[prev_idx].close
                change = candle.close - prev_close
                change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                change_str = f"{change:+.2f} ({change_pct:+.2f}%)"
                change_style = "green" if change >= 0 else "red"
            else:
                change_str = "-"
                change_style = "dim"
        else:
            change_str = "-"
            change_style = "dim"
        
        # Format timestamp based on timeframe
        if timeframe == "1day":
            ts_str = candle.timestamp.strftime("%Y-%m-%d")
        else:
            ts_str = candle.timestamp.strftime("%Y-%m-%d %H:%M")
        
        table.add_row(
            ts_str,
            f"{candle.open:.2f}",
            f"{candle.high:.2f}",
            f"{candle.low:.2f}",
            f"{candle.close:.2f}",
            f"{candle.volume:,}",
            f"[{change_style}]{change_str}[/{change_style}]",
        )
    
    console.print(table)
    
    if len(candles) > 20:
        console.print(f"[dim]Showing last 20 of {len(candles)} candles[/dim]")


@click.command()
@click.argument("symbols", nargs=-1, required=True)
@click.option(
    "-r", "--refresh",
    default=1,
    type=int,
    help="Refresh interval in seconds (default: 1)",
)
def live(symbols: tuple[str, ...], refresh: int) -> None:
    """Watch live prices for one or more symbols.
    
    SYMBOLS are the trading symbols to watch (e.g., RELIANCE INFY TCS).
    
    Press Ctrl+C to stop watching.
    
    \b
    Examples:
      daytrader live RELIANCE
      daytrader live YESBANK IDEA SBIN
      daytrader live RELIANCE --refresh 2
    """
    import time
    from rich.live import Live
    
    config = _get_config()
    
    if config is None:
        console.print(Panel(
            "[red]Configuration not found.[/red]\n\n"
            "Run [cyan]daytrader login[/cyan] to create a config file.",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    symbols = tuple(s.upper() for s in symbols)
    
    try:
        broker = _get_broker(config)
        
        def generate_table() -> Table:
            """Generate the live price table."""
            table = Table(
                title="Live Prices (Ctrl+C to stop)",
                show_header=True,
                header_style="bold cyan",
            )
            
            table.add_column("Symbol", style="bold")
            table.add_column("LTP", justify="right")
            table.add_column("Change", justify="right")
            table.add_column("Change %", justify="right")
            table.add_column("Open", justify="right", style="dim")
            table.add_column("High", justify="right", style="green")
            table.add_column("Low", justify="right", style="red")
            
            for symbol in symbols:
                try:
                    q = broker.get_quote(symbol)
                    
                    if q.change >= 0:
                        change_color = "green"
                        arrow = "▲"
                    else:
                        change_color = "red"
                        arrow = "▼"
                    
                    table.add_row(
                        symbol,
                        f"₹{q.ltp:.2f}",
                        f"[{change_color}]{arrow} {q.change:+.2f}[/{change_color}]",
                        f"[{change_color}]{q.change_percent:+.2f}%[/{change_color}]",
                        f"₹{q.open:.2f}",
                        f"₹{q.high:.2f}",
                        f"₹{q.low:.2f}",
                    )
                except Exception as e:
                    table.add_row(
                        symbol,
                        "[red]Error[/red]",
                        "-",
                        "-",
                        "-",
                        "-",
                        "-",
                    )
            
            return table
        
        console.print(f"[dim]Watching {len(symbols)} symbol(s), refreshing every {refresh}s...[/dim]\n")
        
        with Live(generate_table(), refresh_per_second=1, console=console) as live_display:
            while True:
                time.sleep(refresh)
                live_display.update(generate_table())
                
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped watching.[/dim]")
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to watch prices:[/red]\n\n{str(e)}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)


@click.command()
@click.argument("symbol")
def quote(symbol: str) -> None:
    """Display current quote for a symbol.
    
    SYMBOL is the trading symbol (e.g., RELIANCE, INFY, TCS).
    
    Shows LTP (Last Traded Price), change, and volume.
    
    \b
    Examples:
      daytrader quote RELIANCE
      daytrader quote INFY
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
    
    try:
        broker = _get_broker(config)
        q = broker.get_quote(symbol)
        
        # Determine color based on change
        if q.change >= 0:
            change_color = "green"
            arrow = "▲"
        else:
            change_color = "red"
            arrow = "▼"
        
        # Build quote display
        quote_text = (
            f"[bold]{symbol}[/bold]\n\n"
            f"[bold white]LTP:[/bold white] ₹{q.ltp:.2f}\n"
            f"[bold white]Change:[/bold white] [{change_color}]{arrow} {q.change:+.2f} ({q.change_percent:+.2f}%)[/{change_color}]\n\n"
            f"[dim]Open:[/dim]   ₹{q.open:.2f}\n"
            f"[dim]High:[/dim]   ₹{q.high:.2f}\n"
            f"[dim]Low:[/dim]    ₹{q.low:.2f}\n"
            f"[dim]Close:[/dim]  ₹{q.close:.2f}\n"
            f"[dim]Volume:[/dim] {q.volume:,}"
        )
        
        console.print(Panel(
            quote_text,
            title=f"[bold {change_color}]Quote[/bold {change_color}]",
            border_style=change_color,
        ))
        
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to get quote:[/red]\n\n{str(e)}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
