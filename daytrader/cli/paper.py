"""Paper trading commands for DayTrader CLI.

Handles paper trading management including reset and status commands.
"""

from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _get_config():
    """Lazily load configuration."""
    import toml
    
    config_path = Path.home() / ".config" / "daytrader" / "config.toml"
    
    if not config_path.exists():
        return None
    
    try:
        return toml.load(config_path)
    except Exception:
        return None


def _get_db_path() -> Path:
    """Get the database path."""
    return Path.home() / ".config" / "daytrader" / "daytrader.db"


def _get_data_store():
    """Get the data store instance."""
    from daytrader.db.store import DataStore
    
    return DataStore(_get_db_path())


def _get_paper_broker():
    """Get the paper broker instance."""
    from daytrader.brokers.paper import PaperBroker
    
    config = _get_config()
    starting_balance = 100000.0
    
    if config:
        starting_balance = config.get("trading", {}).get("paper_starting_balance", 100000.0)
    
    store = _get_data_store()
    return PaperBroker(data_store=store, starting_balance=starting_balance)


def _is_paper_mode() -> bool:
    """Check if trading mode is set to paper."""
    config = _get_config()
    if config is None:
        return True  # Default to paper mode if no config
    
    return config.get("trading", {}).get("mode", "paper") == "paper"


@click.group()
def paper() -> None:
    """Paper trading management commands.
    
    Manage your paper trading account including resetting balance
    and viewing current status.
    
    \b
    Commands:
      reset   - Reset paper trading to initial state
      status  - View paper trading account status
    """
    pass


@paper.command()
@click.option(
    "--confirm",
    is_flag=True,
    help="Skip confirmation prompt.",
)
def reset(confirm: bool) -> None:
    """Reset paper trading to initial state.
    
    Clears all paper trading positions and resets the balance
    to the configured starting amount.
    
    \b
    Examples:
      daytrader paper reset           # Reset with confirmation
      daytrader paper reset --confirm # Reset without confirmation
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
    
    # Get starting balance from config
    starting_balance = config.get("trading", {}).get("paper_starting_balance", 100000.0)
    
    # Get current state before reset
    broker = _get_paper_broker()
    current_balance = broker.get_current_balance()
    positions = broker.get_positions()
    
    # Show current state
    console.print("[bold cyan]Paper Trading Reset[/bold cyan]\n")
    console.print(f"Current Balance: [yellow]₹{current_balance:,.2f}[/yellow]")
    console.print(f"Open Positions:  [yellow]{len(positions)}[/yellow]")
    console.print(f"Starting Balance: [green]₹{starting_balance:,.2f}[/green]\n")
    
    if not confirm:
        if not click.confirm("Are you sure you want to reset paper trading?"):
            console.print("[dim]Reset cancelled.[/dim]")
            return
    
    # Perform reset
    broker.reset()
    
    console.print(Panel(
        f"[green]Paper trading has been reset![/green]\n\n"
        f"Balance: ₹{starting_balance:,.2f}\n"
        f"Positions: 0",
        title="[bold green]Reset Complete[/bold green]",
        border_style="green",
    ))


@paper.command()
def status() -> None:
    """View paper trading account status.
    
    Displays current balance, positions, and P&L for paper trading.
    
    \b
    Examples:
      daytrader paper status
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
    
    # Check if in paper mode
    trading_mode = config.get("trading", {}).get("mode", "paper")
    starting_balance = config.get("trading", {}).get("paper_starting_balance", 100000.0)
    
    broker = _get_paper_broker()
    balance_info = broker.get_balance()
    positions = broker.get_positions()
    
    # Calculate total P&L
    total_unrealized_pnl = sum(p.pnl for p in positions)
    total_value = balance_info.total_value
    overall_pnl = total_value - starting_balance
    overall_pnl_percent = (overall_pnl / starting_balance * 100) if starting_balance > 0 else 0
    
    # Display header
    mode_color = "green" if trading_mode == "paper" else "red"
    console.print(f"[bold cyan]Paper Trading Status[/bold cyan] [{mode_color}]({trading_mode.upper()} MODE)[/{mode_color}]\n")
    
    # Account summary
    pnl_color = "green" if overall_pnl >= 0 else "red"
    pnl_sign = "+" if overall_pnl >= 0 else ""
    
    summary_text = (
        f"[bold]Account Summary[/bold]\n\n"
        f"Starting Balance:   ₹{starting_balance:,.2f}\n"
        f"Available Cash:     ₹{balance_info.available_cash:,.2f}\n"
        f"Used Margin:        ₹{balance_info.used_margin:,.2f}\n"
        f"Total Value:        ₹{total_value:,.2f}\n"
        f"{'─' * 35}\n"
        f"Overall P&L:        [{pnl_color}]{pnl_sign}₹{overall_pnl:,.2f} ({pnl_sign}{overall_pnl_percent:.2f}%)[/{pnl_color}]"
    )
    
    console.print(Panel(
        summary_text,
        title="[bold]Account[/bold]",
        border_style="cyan",
    ))
    
    # Positions table
    if positions:
        console.print()
        table = Table(
            title="Open Positions",
            show_header=True,
            header_style="bold",
        )
        
        table.add_column("Symbol", style="bold")
        table.add_column("Qty", justify="right")
        table.add_column("Avg Price", justify="right")
        table.add_column("LTP", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")
        table.add_column("Product", justify="center")
        
        for pos in positions:
            pnl_color = "green" if pos.pnl >= 0 else "red"
            pnl_sign = "+" if pos.pnl >= 0 else ""
            
            table.add_row(
                pos.symbol,
                str(pos.quantity),
                f"₹{pos.average_price:.2f}",
                f"₹{pos.ltp:.2f}",
                f"[{pnl_color}]{pnl_sign}₹{pos.pnl:.2f}[/{pnl_color}]",
                f"[{pnl_color}]{pnl_sign}{pos.pnl_percent:.2f}%[/{pnl_color}]",
                pos.product,
            )
        
        console.print(table)
        
        # Unrealized P&L summary
        unrealized_color = "green" if total_unrealized_pnl >= 0 else "red"
        unrealized_sign = "+" if total_unrealized_pnl >= 0 else ""
        console.print(f"\nUnrealized P&L: [{unrealized_color}]{unrealized_sign}₹{total_unrealized_pnl:,.2f}[/{unrealized_color}]")
    else:
        console.print("\n[dim]No open positions.[/dim]")
    
    # Tips
    console.print(Panel(
        "[dim]Commands:[/dim]\n"
        "• [cyan]daytrader buy SYMBOL QTY[/cyan] - Buy stocks\n"
        "• [cyan]daytrader sell SYMBOL QTY[/cyan] - Sell stocks\n"
        "• [cyan]daytrader paper reset[/cyan] - Reset paper trading",
        title="[bold]Quick Actions[/bold]",
        border_style="dim",
    ))
