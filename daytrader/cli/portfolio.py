"""Portfolio commands for DayTrader CLI.

Handles P&L display, balance information, and trade journal.
"""

from datetime import date
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


def calculate_pnl_from_trades(trades: list) -> dict:
    """Calculate P&L metrics from a list of trades.
    
    Args:
        trades: List of Trade objects.
        
    Returns:
        Dictionary with P&L metrics.
    """
    if not trades:
        return {
            "realized_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
        }
    
    realized_pnl = 0.0
    winning_trades = 0
    losing_trades = 0
    total_wins = 0.0
    total_losses = 0.0
    
    for trade in trades:
        if trade.pnl is not None:
            realized_pnl += trade.pnl
            if trade.pnl > 0:
                winning_trades += 1
                total_wins += trade.pnl
            elif trade.pnl < 0:
                losing_trades += 1
                total_losses += abs(trade.pnl)
    
    total_trades = len(trades)
    trades_with_pnl = winning_trades + losing_trades
    
    win_rate = (winning_trades / trades_with_pnl * 100) if trades_with_pnl > 0 else 0.0
    avg_win = (total_wins / winning_trades) if winning_trades > 0 else 0.0
    avg_loss = (total_losses / losing_trades) if losing_trades > 0 else 0.0
    
    return {
        "realized_pnl": realized_pnl,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


@click.command()
@click.option(
    "--history",
    is_flag=True,
    default=False,
    help="Display P&L history from journal.",
)
def pnl(history: bool) -> None:
    """Display today's realized and unrealized P&L.
    
    Shows realized P&L from closed trades and unrealized P&L from
    open positions. Use --history to view historical P&L from journal.
    
    \b
    Examples:
      daytrader pnl            # Today's P&L
      daytrader pnl --history  # Historical P&L
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
    
    store = _get_data_store()
    
    if history:
        # Display P&L history from journal
        journal_entries = store.get_journal()
        
        if not journal_entries:
            console.print(Panel(
                "[dim]No journal entries found[/dim]",
                title="[bold]P&L History[/bold]",
                border_style="dim",
            ))
            return
        
        table = Table(
            title="P&L History",
            show_header=True,
            header_style="bold cyan",
        )
        
        table.add_column("Date", style="bold")
        table.add_column("Trades", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("Win Rate", justify="right")
        table.add_column("Notes", max_width=30)
        
        total_pnl = 0.0
        
        for entry in journal_entries:
            pnl_color = "green" if entry.total_pnl >= 0 else "red"
            pnl_sign = "+" if entry.total_pnl >= 0 else ""
            
            table.add_row(
                entry.date.strftime("%Y-%m-%d"),
                str(entry.trades_count),
                f"[{pnl_color}]{pnl_sign}₹{entry.total_pnl:.2f}[/{pnl_color}]",
                f"{entry.win_rate:.1f}%",
                (entry.notes[:27] + "...") if entry.notes and len(entry.notes) > 30 else (entry.notes or "-"),
            )
            total_pnl += entry.total_pnl
        
        console.print(table)
        
        # Show total
        total_color = "green" if total_pnl >= 0 else "red"
        total_sign = "+" if total_pnl >= 0 else ""
        console.print(f"\n[bold]Total P&L:[/bold] [{total_color}]{total_sign}₹{total_pnl:.2f}[/{total_color}]")
        
    else:
        # Display today's P&L
        today = date.today()
        
        # Get today's trades for realized P&L
        trades = store.get_trades(trade_date=today)
        pnl_metrics = calculate_pnl_from_trades(trades)
        
        # Get unrealized P&L from positions
        try:
            broker = _get_broker(config)
            positions = broker.get_positions()
            unrealized_pnl = sum(pos.pnl for pos in positions)
        except Exception:
            positions = []
            unrealized_pnl = 0.0
        
        total_pnl = pnl_metrics["realized_pnl"] + unrealized_pnl
        
        # Build P&L display
        realized_color = "green" if pnl_metrics["realized_pnl"] >= 0 else "red"
        realized_sign = "+" if pnl_metrics["realized_pnl"] >= 0 else ""
        
        unrealized_color = "green" if unrealized_pnl >= 0 else "red"
        unrealized_sign = "+" if unrealized_pnl >= 0 else ""
        
        total_color = "green" if total_pnl >= 0 else "red"
        total_sign = "+" if total_pnl >= 0 else ""
        
        pnl_text = (
            f"[bold]Today's P&L Summary[/bold] ({today.strftime('%Y-%m-%d')})\n\n"
            f"Realized P&L:   [{realized_color}]{realized_sign}₹{pnl_metrics['realized_pnl']:.2f}[/{realized_color}]\n"
            f"Unrealized P&L: [{unrealized_color}]{unrealized_sign}₹{unrealized_pnl:.2f}[/{unrealized_color}]\n"
            f"{'─' * 30}\n"
            f"[bold]Total P&L:      [{total_color}]{total_sign}₹{total_pnl:.2f}[/{total_color}][/bold]\n\n"
            f"[dim]Trades: {pnl_metrics['total_trades']} | "
            f"Wins: {pnl_metrics['winning_trades']} | "
            f"Losses: {pnl_metrics['losing_trades']} | "
            f"Win Rate: {pnl_metrics['win_rate']:.1f}%[/dim]"
        )
        
        if pnl_metrics["winning_trades"] > 0 or pnl_metrics["losing_trades"] > 0:
            pnl_text += (
                f"\n[dim]Avg Win: ₹{pnl_metrics['avg_win']:.2f} | "
                f"Avg Loss: ₹{pnl_metrics['avg_loss']:.2f}[/dim]"
            )
        
        console.print(Panel(
            pnl_text,
            title="[bold cyan]P&L[/bold cyan]",
            border_style="cyan",
        ))
        
        # Show open positions if any
        if positions:
            console.print(f"\n[dim]Open positions: {len(positions)}[/dim]")


@click.command()
def balance() -> None:
    """Display available margin and funds.
    
    Shows available cash, used margin, and total portfolio value.
    
    \b
    Examples:
      daytrader balance
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
    
    try:
        broker = _get_broker(config)
        bal = broker.get_balance()
        
        trading_mode = config.get("trading", {}).get("mode", "paper")
        mode_label = "[yellow](Paper)[/yellow]" if trading_mode == "paper" else "[green](Live)[/green]"
        
        balance_text = (
            f"[bold]Account Balance[/bold] {mode_label}\n\n"
            f"Available Cash:  [green]₹{bal.available_cash:,.2f}[/green]\n"
            f"Used Margin:     [yellow]₹{bal.used_margin:,.2f}[/yellow]\n"
            f"{'─' * 30}\n"
            f"[bold]Total Value:     ₹{bal.total_value:,.2f}[/bold]"
        )
        
        console.print(Panel(
            balance_text,
            title="[bold cyan]Balance[/bold cyan]",
            border_style="cyan",
        ))
        
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to get balance:[/red]\n\n{str(e)}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)


@click.command()
@click.option(
    "--days",
    type=int,
    default=None,
    help="Number of days of history to show.",
)
def journal(days: Optional[int]) -> None:
    """Display trade history with notes.
    
    Shows all trades from the database with timestamps, symbols,
    sides, quantities, prices, and P&L.
    
    \b
    Examples:
      daytrader journal           # All trades
      daytrader journal --days 7  # Last 7 days
    """
    from datetime import timedelta
    
    config = _get_config()
    
    if config is None:
        console.print(Panel(
            "[red]Configuration not found.[/red]\n\n"
            "Run [cyan]daytrader login[/cyan] to create a config file.",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    store = _get_data_store()
    
    # Get trades
    if days is not None:
        from_date = date.today() - timedelta(days=days)
        # Get all trades and filter by date
        all_trades = store.get_trades()
        trades = [t for t in all_trades if t.timestamp.date() >= from_date]
    else:
        trades = store.get_trades()
    
    if not trades:
        console.print(Panel(
            "[dim]No trades found[/dim]",
            title="[bold]Trade Journal[/bold]",
            border_style="dim",
        ))
        return
    
    # Create trades table
    table = Table(
        title="Trade Journal",
        show_header=True,
        header_style="bold cyan",
    )
    
    table.add_column("Date/Time", style="dim")
    table.add_column("Symbol", style="bold")
    table.add_column("Side", justify="center")
    table.add_column("Qty", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Mode", justify="center", style="dim")
    
    total_pnl = 0.0
    
    for trade in trades:
        side_color = "green" if trade.side == "BUY" else "red"
        
        if trade.pnl is not None:
            pnl_color = "green" if trade.pnl >= 0 else "red"
            pnl_sign = "+" if trade.pnl >= 0 else ""
            pnl_str = f"[{pnl_color}]{pnl_sign}₹{trade.pnl:.2f}[/{pnl_color}]"
            total_pnl += trade.pnl
        else:
            pnl_str = "-"
        
        mode_str = "[yellow]Paper[/yellow]" if trade.is_paper else "[green]Live[/green]"
        
        table.add_row(
            trade.timestamp.strftime("%Y-%m-%d %H:%M"),
            trade.symbol,
            f"[{side_color}]{trade.side}[/{side_color}]",
            str(trade.quantity),
            f"₹{trade.price:.2f}",
            pnl_str,
            mode_str,
        )
    
    console.print(table)
    
    # Show summary
    total_color = "green" if total_pnl >= 0 else "red"
    total_sign = "+" if total_pnl >= 0 else ""
    console.print(f"\n[bold]Total Trades:[/bold] {len(trades)}")
    console.print(f"[bold]Total P&L:[/bold] [{total_color}]{total_sign}₹{total_pnl:.2f}[/{total_color}]")


@click.command()
@click.option(
    "--date",
    "trade_date",
    type=str,
    default=None,
    help="Date to sync (YYYY-MM-DD). Defaults to today.",
)
def sync(trade_date: Optional[str]) -> None:
    """Sync trades from Angel One to local journal.
    
    Fetches order book and trade book from Angel One and imports
    executed trades into the local database for tracking and analysis.
    
    \b
    Examples:
      daytrader sync              # Sync today's trades
      daytrader sync --date 2024-12-19  # Sync specific date
    """
    from datetime import datetime
    from daytrader.brokers.angelone import AngelOneBroker
    from daytrader.models import Trade
    
    config = _get_config()
    
    if config is None:
        console.print(Panel(
            "[red]Configuration not found.[/red]\n\n"
            "Run [cyan]daytrader login[/cyan] to create a config file.",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    angelone_config = config.get("angelone", {})
    if not angelone_config.get("api_key"):
        console.print(Panel(
            "[yellow]Angel One not configured.[/yellow]",
            title="[bold yellow]Configuration Required[/bold yellow]",
            border_style="yellow",
        ))
        return
    
    # Parse date
    if trade_date:
        try:
            sync_date = date.fromisoformat(trade_date)
        except ValueError:
            console.print(f"[red]Invalid date format: {trade_date}. Use YYYY-MM-DD[/red]")
            return
    else:
        sync_date = date.today()
    
    console.print(f"[dim]Syncing trades for {sync_date}...[/dim]")
    
    try:
        broker = AngelOneBroker(
            api_key=angelone_config.get("api_key", ""),
            client_id=angelone_config.get("client_id", ""),
            pin=angelone_config.get("pin", ""),
            totp_secret=angelone_config.get("totp_secret", ""),
        )
        broker._ensure_authenticated()
        
        # Get trade book (executed trades)
        trade_book = broker._smart_api.tradeBook()
        
        if not trade_book or not trade_book.get("data"):
            console.print(Panel(
                f"[dim]No trades found for {sync_date}[/dim]",
                title="[bold]Sync Complete[/bold]",
                border_style="dim",
            ))
            return
        
        store = _get_data_store()
        existing_trades = store.get_trades(trade_date=sync_date)
        existing_order_ids = {t.order_id for t in existing_trades}
        
        imported = 0
        skipped = 0
        total_buy_value = 0.0
        total_sell_value = 0.0
        
        for trade_data in trade_book["data"]:
            order_id = trade_data.get("orderid", "")
            
            # Skip if already imported
            if order_id in existing_order_ids:
                skipped += 1
                continue
            
            # Parse trade data
            symbol = trade_data.get("tradingsymbol", "")
            side = trade_data.get("transactiontype", "")
            quantity = int(trade_data.get("fillsize", 0) or trade_data.get("quantity", 0))
            price = float(trade_data.get("fillprice", 0) or trade_data.get("averageprice", 0))
            
            # Parse timestamp - filltime is just HH:MM:SS, need to add date
            fill_time = trade_data.get("filltime", "")
            try:
                if fill_time and ":" in fill_time:
                    # filltime is just "HH:MM:SS", combine with sync_date
                    time_parts = fill_time.split(":")
                    timestamp = datetime.combine(
                        sync_date,
                        datetime.strptime(fill_time, "%H:%M:%S").time()
                    )
                else:
                    timestamp = datetime.combine(sync_date, datetime.now().time())
            except ValueError:
                timestamp = datetime.combine(sync_date, datetime.now().time())
            
            if quantity > 0 and price > 0:
                trade = Trade(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=price,
                    timestamp=timestamp,
                    order_id=order_id,
                    is_paper=False,
                    pnl=None,
                )
                store.log_trade(trade)
                imported += 1
                
                if side == "BUY":
                    total_buy_value += price * quantity
                else:
                    total_sell_value += price * quantity
        
        # Build summary
        net_value = total_sell_value - total_buy_value
        net_color = "green" if net_value >= 0 else "red"
        
        summary = (
            f"[bold]Sync Complete[/bold]\n\n"
            f"Date:     {sync_date}\n"
            f"Imported: {imported} trades\n"
            f"Skipped:  {skipped} (already in journal)\n\n"
            f"Buy Value:  ₹{total_buy_value:,.2f}\n"
            f"Sell Value: ₹{total_sell_value:,.2f}\n"
            f"Net:        [{net_color}]₹{net_value:+,.2f}[/{net_color}]"
        )
        
        console.print(Panel(
            summary,
            title="[bold cyan]Trade Sync[/bold cyan]",
            border_style="cyan",
        ))
        
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to sync trades:[/red]\n\n{str(e)}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)


@click.command()
@click.option(
    "--days",
    type=int,
    default=7,
    help="Number of days to analyze (default: 7).",
)
def report(days: int) -> None:
    """Generate trading performance report with analysis.
    
    Analyzes your trading history and provides insights on:
    - Win rate and profit factor
    - Best and worst trades
    - Average holding time
    - Most traded symbols
    - Daily P&L breakdown
    
    \b
    Examples:
      daytrader report           # Last 7 days
      daytrader report --days 30 # Last 30 days
    """
    from datetime import timedelta
    from collections import defaultdict
    
    config = _get_config()
    
    if config is None:
        console.print(Panel(
            "[red]Configuration not found.[/red]\n\n"
            "Run [cyan]daytrader login[/cyan] to create a config file.",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    store = _get_data_store()
    
    # Get trades for the period
    from_date = date.today() - timedelta(days=days)
    all_trades = store.get_trades()
    trades = [t for t in all_trades if t.timestamp.date() >= from_date]
    
    if not trades:
        console.print(Panel(
            f"[dim]No trades found in the last {days} days[/dim]\n\n"
            "[dim]Run 'daytrader sync' to import trades from Angel One[/dim]",
            title="[bold]Trading Report[/bold]",
            border_style="dim",
        ))
        return
    
    # Calculate metrics
    buy_trades = [t for t in trades if t.side == "BUY"]
    sell_trades = [t for t in trades if t.side == "SELL"]
    
    total_buy_value = sum(t.price * t.quantity for t in buy_trades)
    total_sell_value = sum(t.price * t.quantity for t in sell_trades)
    
    # Group by symbol
    symbol_stats = defaultdict(lambda: {"buys": 0, "sells": 0, "buy_value": 0, "sell_value": 0})
    for t in trades:
        symbol_stats[t.symbol]["buys" if t.side == "BUY" else "sells"] += t.quantity
        symbol_stats[t.symbol]["buy_value" if t.side == "BUY" else "sell_value"] += t.price * t.quantity
    
    # Group by date
    daily_stats = defaultdict(lambda: {"trades": 0, "buy_value": 0, "sell_value": 0})
    for t in trades:
        day = t.timestamp.date().isoformat()
        daily_stats[day]["trades"] += 1
        if t.side == "BUY":
            daily_stats[day]["buy_value"] += t.price * t.quantity
        else:
            daily_stats[day]["sell_value"] += t.price * t.quantity
    
    # Calculate P&L per symbol (simplified - assumes FIFO)
    symbol_pnl = {}
    for symbol, stats in symbol_stats.items():
        if stats["sells"] > 0 and stats["buys"] > 0:
            avg_buy = stats["buy_value"] / stats["buys"] if stats["buys"] > 0 else 0
            avg_sell = stats["sell_value"] / stats["sells"] if stats["sells"] > 0 else 0
            closed_qty = min(stats["buys"], stats["sells"])
            pnl = (avg_sell - avg_buy) * closed_qty
            symbol_pnl[symbol] = pnl
    
    total_realized_pnl = sum(symbol_pnl.values())
    
    # Find best and worst
    if symbol_pnl:
        best_symbol = max(symbol_pnl, key=symbol_pnl.get)
        worst_symbol = min(symbol_pnl, key=symbol_pnl.get)
    else:
        best_symbol = worst_symbol = None
    
    # Build report
    output_lines = [
        f"[bold]Trading Report[/bold] - Last {days} days\n",
        f"[bold]Overview:[/bold]",
        f"  Total Trades:  {len(trades)}",
        f"  Buy Orders:    {len(buy_trades)}",
        f"  Sell Orders:   {len(sell_trades)}",
        f"  Trading Days:  {len(daily_stats)}\n",
        f"[bold]Value:[/bold]",
        f"  Total Bought:  ₹{total_buy_value:,.2f}",
        f"  Total Sold:    ₹{total_sell_value:,.2f}",
    ]
    
    if total_realized_pnl != 0:
        pnl_color = "green" if total_realized_pnl >= 0 else "red"
        output_lines.append(f"  Realized P&L:  [{pnl_color}]₹{total_realized_pnl:+,.2f}[/{pnl_color}]")
    
    # Most traded symbols
    output_lines.append(f"\n[bold]Most Traded Symbols:[/bold]")
    sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]["buys"] + x[1]["sells"], reverse=True)[:5]
    for symbol, stats in sorted_symbols:
        total_qty = stats["buys"] + stats["sells"]
        pnl = symbol_pnl.get(symbol, 0)
        pnl_str = f"[{'green' if pnl >= 0 else 'red'}]₹{pnl:+,.2f}[/]" if pnl != 0 else ""
        output_lines.append(f"  {symbol}: {total_qty} shares {pnl_str}")
    
    # Best/Worst trades
    if best_symbol and symbol_pnl[best_symbol] > 0:
        output_lines.append(f"\n[bold]Best Trade:[/bold] [green]{best_symbol} +₹{symbol_pnl[best_symbol]:,.2f}[/green]")
    if worst_symbol and symbol_pnl[worst_symbol] < 0:
        output_lines.append(f"[bold]Worst Trade:[/bold] [red]{worst_symbol} ₹{symbol_pnl[worst_symbol]:,.2f}[/red]")
    
    # Daily breakdown
    output_lines.append(f"\n[bold]Daily Breakdown:[/bold]")
    for day in sorted(daily_stats.keys(), reverse=True)[:5]:
        stats = daily_stats[day]
        net = stats["sell_value"] - stats["buy_value"]
        net_color = "green" if net >= 0 else "red"
        output_lines.append(f"  {day}: {stats['trades']} trades, [{net_color}]₹{net:+,.2f}[/{net_color}]")
    
    # Trading insights
    output_lines.append(f"\n[bold]Insights:[/bold]")
    
    avg_trade_value = (total_buy_value + total_sell_value) / len(trades) if trades else 0
    output_lines.append(f"  Avg Trade Size: ₹{avg_trade_value:,.2f}")
    
    if len(daily_stats) > 0:
        avg_trades_per_day = len(trades) / len(daily_stats)
        output_lines.append(f"  Avg Trades/Day: {avg_trades_per_day:.1f}")
    
    # Win rate (if we have closed trades)
    if symbol_pnl:
        winners = sum(1 for p in symbol_pnl.values() if p > 0)
        losers = sum(1 for p in symbol_pnl.values() if p < 0)
        if winners + losers > 0:
            win_rate = winners / (winners + losers) * 100
            output_lines.append(f"  Win Rate: {win_rate:.1f}% ({winners}W / {losers}L)")
    
    console.print(Panel(
        "\n".join(output_lines),
        title="[bold cyan]Performance Report[/bold cyan]",
        border_style="cyan",
    ))
