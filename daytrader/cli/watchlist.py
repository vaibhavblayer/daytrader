"""Watchlist management commands for DayTrader CLI.

Handles watchlist operations including add, remove, list, and import commands.
Supports multiple named watchlists.
"""

from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


# Predefined index constituents for import
PREDEFINED_LISTS = {
    "nifty50": [
        "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
        "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
        "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
        "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
        "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
        "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
        "M&M", "MARUTI", "NTPC", "NESTLEIND", "ONGC",
        "POWERGRID", "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA",
        "TCS", "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TECHM",
        "TITAN", "ULTRACEMCO", "UPL", "WIPRO", "LTIM",
    ],
    "banknifty": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
        "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "PNB",
        "AUBANK", "BANKBARODA",
    ],
    "niftyit": [
        "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
        "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "LTTS",
    ],
}


def _get_data_store():
    """Get the data store instance."""
    from pathlib import Path
    from daytrader.db.store import DataStore
    
    db_path = Path.home() / ".config" / "daytrader" / "daytrader.db"
    return DataStore(db_path)


@click.group()
def watch() -> None:
    """Manage watchlists.
    
    Add, remove, and view symbols in your watchlists.
    Supports multiple named watchlists for organizing stocks.
    
    \b
    Examples:
      daytrader watch add RELIANCE          # Add to default watchlist
      daytrader watch add INFY --list tech  # Add to 'tech' watchlist
      daytrader watch list                  # Show default watchlist
      daytrader watch list --list tech      # Show 'tech' watchlist
      daytrader watch import nifty50        # Import Nifty 50 stocks
    """
    pass


@watch.command("add")
@click.argument("symbol")
@click.option(
    "--list", "list_name",
    default="default",
    help="Name of the watchlist to add to (default: 'default').",
)
def add_symbol(symbol: str, list_name: str) -> None:
    """Add a symbol to a watchlist.
    
    SYMBOL is the trading symbol to add (e.g., RELIANCE, INFY, TCS).
    
    \b
    Examples:
      daytrader watch add RELIANCE
      daytrader watch add INFY --list tech
    """
    symbol = symbol.upper()
    
    try:
        store = _get_data_store()
        
        # Check if already in watchlist
        current = store.get_watchlist(list_name)
        if symbol in current:
            console.print(f"[yellow]{symbol} is already in watchlist '{list_name}'[/yellow]")
            return
        
        store.add_to_watchlist(symbol, list_name)
        console.print(f"[green]✓ Added {symbol} to watchlist '{list_name}'[/green]")
        
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to add symbol:[/red]\n\n{str(e)}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)


@watch.command("remove")
@click.argument("symbol")
@click.option(
    "--list", "list_name",
    default="default",
    help="Name of the watchlist to remove from (default: 'default').",
)
def remove_symbol(symbol: str, list_name: str) -> None:
    """Remove a symbol from a watchlist.
    
    SYMBOL is the trading symbol to remove.
    
    \b
    Examples:
      daytrader watch remove RELIANCE
      daytrader watch remove INFY --list tech
    """
    symbol = symbol.upper()
    
    try:
        store = _get_data_store()
        
        # Check if in watchlist
        current = store.get_watchlist(list_name)
        if symbol not in current:
            console.print(f"[yellow]{symbol} is not in watchlist '{list_name}'[/yellow]")
            return
        
        store.remove_from_watchlist(symbol, list_name)
        console.print(f"[green]✓ Removed {symbol} from watchlist '{list_name}'[/green]")
        
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to remove symbol:[/red]\n\n{str(e)}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)


@watch.command("list")
@click.option(
    "--list", "list_name",
    default=None,
    help="Name of the watchlist to display. If not specified, shows all watchlists.",
)
def list_watchlist(list_name: Optional[str]) -> None:
    """Display watchlist symbols.
    
    Shows all symbols in the specified watchlist, or all watchlists if
    no list name is provided.
    
    \b
    Examples:
      daytrader watch list                  # Show all watchlists
      daytrader watch list --list default   # Show default watchlist
      daytrader watch list --list tech      # Show 'tech' watchlist
    """
    try:
        store = _get_data_store()
        
        if list_name:
            # Show specific watchlist
            symbols = store.get_watchlist(list_name)
            
            if not symbols:
                console.print(Panel(
                    f"[dim]Watchlist '{list_name}' is empty[/dim]",
                    title=f"[bold]Watchlist: {list_name}[/bold]",
                    border_style="dim",
                ))
                return
            
            table = Table(
                title=f"Watchlist: {list_name}",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("#", style="dim", width=4)
            table.add_column("Symbol", style="bold")
            
            for i, symbol in enumerate(sorted(symbols), 1):
                table.add_row(str(i), symbol)
            
            console.print(table)
            console.print(f"\n[dim]Total: {len(symbols)} symbols[/dim]")
        else:
            # Show all watchlists
            watchlist_names = store.get_watchlist_names()
            
            if not watchlist_names:
                console.print(Panel(
                    "[dim]No watchlists found. Use 'daytrader watch add SYMBOL' to create one.[/dim]",
                    title="[bold]Watchlists[/bold]",
                    border_style="dim",
                ))
                return
            
            for wl_name in sorted(watchlist_names):
                symbols = store.get_watchlist(wl_name)
                
                table = Table(
                    title=f"Watchlist: {wl_name}",
                    show_header=True,
                    header_style="bold cyan",
                )
                table.add_column("#", style="dim", width=4)
                table.add_column("Symbol", style="bold")
                
                for i, symbol in enumerate(sorted(symbols), 1):
                    table.add_row(str(i), symbol)
                
                console.print(table)
                console.print(f"[dim]Total: {len(symbols)} symbols[/dim]\n")
        
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to list watchlist:[/red]\n\n{str(e)}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)


@watch.command("import")
@click.argument("name")
@click.option(
    "--list", "list_name",
    default=None,
    help="Name of the watchlist to import into. Defaults to the import name.",
)
def import_list(name: str, list_name: Optional[str]) -> None:
    """Import predefined index constituents.
    
    NAME is the predefined list to import (nifty50, banknifty, niftyit).
    
    \b
    Available lists:
      nifty50    - Nifty 50 index constituents
      banknifty  - Bank Nifty index constituents
      niftyit    - Nifty IT index constituents
    
    \b
    Examples:
      daytrader watch import nifty50
      daytrader watch import banknifty --list banks
    """
    name = name.lower()
    
    if name not in PREDEFINED_LISTS:
        available = ", ".join(PREDEFINED_LISTS.keys())
        console.print(Panel(
            f"[red]Unknown list: {name}[/red]\n\n"
            f"Available lists: {available}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    target_list = list_name or name
    symbols = PREDEFINED_LISTS[name]
    
    try:
        store = _get_data_store()
        
        added_count = 0
        skipped_count = 0
        current = store.get_watchlist(target_list)
        
        with console.status(f"[bold cyan]Importing {name}...[/bold cyan]"):
            for symbol in symbols:
                if symbol in current:
                    skipped_count += 1
                else:
                    store.add_to_watchlist(symbol, target_list)
                    added_count += 1
        
        console.print(Panel(
            f"[bold green]Import Complete[/bold green]\n\n"
            f"Added:   {added_count} symbols\n"
            f"Skipped: {skipped_count} (already in list)\n"
            f"Total:   {len(symbols)} symbols in '{target_list}'",
            title=f"[bold]Imported: {name}[/bold]",
            border_style="green",
        ))
        
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to import list:[/red]\n\n{str(e)}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
