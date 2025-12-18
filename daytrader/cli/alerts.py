"""Alert management commands for DayTrader CLI.

Handles alert operations including creating, listing, and removing alerts.
Supports price and indicator conditions.
"""

import re
from datetime import datetime
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from daytrader.models import Alert

console = Console()


# Supported alert condition patterns
CONDITION_PATTERNS = {
    "price_above": re.compile(r"price\s*>\s*([\d.]+)", re.IGNORECASE),
    "price_below": re.compile(r"price\s*<\s*([\d.]+)", re.IGNORECASE),
    "price_gte": re.compile(r"price\s*>=\s*([\d.]+)", re.IGNORECASE),
    "price_lte": re.compile(r"price\s*<=\s*([\d.]+)", re.IGNORECASE),
    "rsi_above": re.compile(r"rsi\s*>\s*([\d.]+)", re.IGNORECASE),
    "rsi_below": re.compile(r"rsi\s*<\s*([\d.]+)", re.IGNORECASE),
    "volume_spike": re.compile(r"volume\s*spike", re.IGNORECASE),
}


def _get_data_store():
    """Get the data store instance."""
    from pathlib import Path
    from daytrader.db.store import DataStore
    
    db_path = Path.home() / ".config" / "daytrader" / "daytrader.db"
    return DataStore(db_path)


def validate_condition(condition: str) -> bool:
    """Validate that a condition string is supported.
    
    Args:
        condition: The condition string to validate.
        
    Returns:
        True if the condition is valid, False otherwise.
    """
    for pattern in CONDITION_PATTERNS.values():
        if pattern.search(condition):
            return True
    return False


def parse_condition(condition: str) -> dict:
    """Parse a condition string into its components.
    
    Args:
        condition: The condition string to parse.
        
    Returns:
        Dictionary with condition type and value.
    """
    for cond_type, pattern in CONDITION_PATTERNS.items():
        match = pattern.search(condition)
        if match:
            if cond_type == "volume_spike":
                return {"type": cond_type, "value": None}
            return {"type": cond_type, "value": float(match.group(1))}
    return {"type": "unknown", "value": None}


def evaluate_condition(condition: str, current_price: float, rsi: Optional[float] = None, 
                       volume_ratio: Optional[float] = None) -> bool:
    """Evaluate if an alert condition is met.
    
    Args:
        condition: The condition string.
        current_price: Current price of the symbol.
        rsi: Current RSI value (optional).
        volume_ratio: Current volume / average volume ratio (optional).
        
    Returns:
        True if condition is met, False otherwise.
    """
    parsed = parse_condition(condition)
    cond_type = parsed["type"]
    value = parsed["value"]
    
    if cond_type == "price_above":
        return current_price > value
    elif cond_type == "price_below":
        return current_price < value
    elif cond_type == "price_gte":
        return current_price >= value
    elif cond_type == "price_lte":
        return current_price <= value
    elif cond_type == "rsi_above" and rsi is not None:
        return rsi > value
    elif cond_type == "rsi_below" and rsi is not None:
        return rsi < value
    elif cond_type == "volume_spike" and volume_ratio is not None:
        return volume_ratio >= 2.0
    
    return False


@click.command("alert")
@click.argument("symbol")
@click.argument("condition")
def create_alert(symbol: str, condition: str) -> None:
    """Create a price or indicator alert.
    
    SYMBOL is the trading symbol (e.g., RELIANCE, INFY).
    CONDITION is the alert condition (e.g., "price > 1500", "rsi < 30").
    
    \b
    Supported conditions:
      price > VALUE    - Alert when price goes above VALUE
      price < VALUE    - Alert when price goes below VALUE
      price >= VALUE   - Alert when price reaches or exceeds VALUE
      price <= VALUE   - Alert when price reaches or falls below VALUE
      rsi > VALUE      - Alert when RSI goes above VALUE
      rsi < VALUE      - Alert when RSI goes below VALUE
      volume spike     - Alert when volume exceeds 2x average
    
    \b
    Examples:
      daytrader alert RELIANCE "price > 2500"
      daytrader alert INFY "rsi < 30"
      daytrader alert TCS "volume spike"
    """
    symbol = symbol.upper()
    
    # Validate condition
    if not validate_condition(condition):
        console.print(Panel(
            f"[red]Invalid condition: {condition}[/red]\n\n"
            "[bold]Supported conditions:[/bold]\n"
            "  price > VALUE\n"
            "  price < VALUE\n"
            "  price >= VALUE\n"
            "  price <= VALUE\n"
            "  rsi > VALUE\n"
            "  rsi < VALUE\n"
            "  volume spike",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    try:
        store = _get_data_store()
        
        # Create alert
        alert = Alert(
            symbol=symbol,
            condition=condition,
            created_at=datetime.now(),
            triggered=False,
        )
        
        alert_id = store.save_alert(alert)
        
        parsed = parse_condition(condition)
        condition_desc = f"{parsed['type'].replace('_', ' ')}"
        if parsed['value'] is not None:
            condition_desc += f" {parsed['value']}"
        
        console.print(Panel(
            f"[bold green]Alert Created[/bold green]\n\n"
            f"ID:        {alert_id}\n"
            f"Symbol:    {symbol}\n"
            f"Condition: {condition}\n"
            f"Type:      {condition_desc}",
            title="[bold]New Alert[/bold]",
            border_style="green",
        ))
        
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to create alert:[/red]\n\n{str(e)}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)


@click.command("alerts")
@click.option(
    "--remove", "remove_id",
    type=int,
    default=None,
    help="Remove alert with specified ID.",
)
def list_alerts(remove_id: Optional[int]) -> None:
    """Display or manage alerts.
    
    Shows all active alerts. Use --remove ID to delete an alert.
    
    \b
    Examples:
      daytrader alerts              # List all alerts
      daytrader alerts --remove 5   # Remove alert with ID 5
    """
    try:
        store = _get_data_store()
        
        if remove_id is not None:
            # Remove alert
            alert = store.get_alert_by_id(remove_id)
            if alert is None:
                console.print(f"[yellow]Alert with ID {remove_id} not found[/yellow]")
                return
            
            store.delete_alert(remove_id)
            console.print(f"[green]✓ Removed alert {remove_id} ({alert.symbol}: {alert.condition})[/green]")
            return
        
        # List all alerts
        alerts = store.get_alerts()
        
        if not alerts:
            console.print(Panel(
                "[dim]No alerts set. Use 'daytrader alert SYMBOL CONDITION' to create one.[/dim]",
                title="[bold]Alerts[/bold]",
                border_style="dim",
            ))
            return
        
        table = Table(
            title="Active Alerts",
            show_header=True,
            header_style="bold cyan",
        )
        
        table.add_column("ID", style="dim", width=6)
        table.add_column("Symbol", style="bold")
        table.add_column("Condition")
        table.add_column("Created", style="dim")
        table.add_column("Status", justify="center")
        
        for alert in alerts:
            status = "[green]●[/green]" if not alert.triggered else "[yellow]✓ Triggered[/yellow]"
            created = alert.created_at.strftime("%Y-%m-%d %H:%M")
            
            table.add_row(
                str(alert.id),
                alert.symbol,
                alert.condition,
                created,
                status,
            )
        
        console.print(table)
        console.print(f"\n[dim]Total: {len(alerts)} alerts[/dim]")
        console.print("[dim]Use 'daytrader alerts --remove ID' to delete an alert[/dim]")
        
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to list alerts:[/red]\n\n{str(e)}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
