"""Trading commands for DayTrader CLI.

Handles order execution including buy, sell, positions display,
and position exit commands.
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


def _get_data_store():
    """Get the data store instance."""
    from pathlib import Path
    from daytrader.db.store import DataStore
    
    db_path = Path.home() / ".config" / "daytrader" / "daytrader.db"
    return DataStore(db_path)


def _log_trade(
    symbol: str,
    side: str,
    quantity: int,
    price: float,
    order_id: str,
    is_paper: bool = False,
    pnl: float = None,
) -> None:
    """Log a trade to the journal.
    
    Args:
        symbol: Trading symbol.
        side: BUY or SELL.
        quantity: Number of shares.
        price: Execution price.
        order_id: Broker order ID.
        is_paper: Whether this is a paper trade.
        pnl: P&L for sell trades.
    """
    from datetime import datetime
    from daytrader.models import Trade
    
    store = _get_data_store()
    
    trade = Trade(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        timestamp=datetime.now(),
        order_id=order_id,
        is_paper=is_paper,
        pnl=pnl,
    )
    
    store.log_trade(trade)


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


@click.command()
@click.argument("symbol")
@click.argument("qty", type=int)
@click.option(
    "-p", "--price",
    type=float,
    default=None,
    help="Limit price for LIMIT orders. If not specified, places a MARKET order.",
)
@click.option(
    "-s", "--sl",
    type=float,
    default=None,
    help="Stop-loss trigger price. Places an SL order after buy executes.",
)
@click.option(
    "-t", "--target",
    type=float,
    default=None,
    help="Target price for profit booking (informational only).",
)
@click.option(
    "-d", "--delivery",
    is_flag=True,
    default=False,
    help="Use CNC (delivery) product type instead of MIS (intraday).",
)
@click.option(
    "-a", "--amo",
    is_flag=True,
    default=False,
    help="Place as After Market Order (executes at next market open).",
)
def buy(
    symbol: str,
    qty: int,
    price: Optional[float],
    sl: Optional[float],
    target: Optional[float],
    delivery: bool,
    amo: bool,
) -> None:
    """Place a buy order for a symbol.
    
    SYMBOL is the trading symbol (e.g., RELIANCE, INFY, TCS).
    QTY is the number of shares to buy.
    
    By default, places a MARKET order with MIS (intraday) product type.
    Use --price for LIMIT orders and --delivery for CNC product type.
    Use --amo to place after market hours (executes at next market open).
    
    \b
    Examples:
      daytrader buy RELIANCE 10              # Market buy 10 shares
      daytrader buy INFY 5 --price 1500      # Limit buy at ₹1500
      daytrader buy TCS 10 --sl 3400         # Market buy with stop-loss
      daytrader buy HDFC 5 --delivery        # Delivery (CNC) order
      daytrader buy TCS 1 --delivery --amo   # AMO delivery order
    """
    from daytrader.models import Order
    
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
    product = "CNC" if delivery else "MIS"
    order_type = "LIMIT" if price is not None else "MARKET"
    variety = "AMO" if amo else "NORMAL"
    
    # Create the order
    order = Order(
        symbol=symbol,
        side="BUY",
        quantity=qty,
        order_type=order_type,
        product=product,
        variety=variety,
        price=price,
        trigger_price=None,
    )
    
    # Display order details before execution
    order_info = (
        f"[bold]Order Details[/bold]\n\n"
        f"Symbol:   {symbol}\n"
        f"Side:     [green]BUY[/green]\n"
        f"Quantity: {qty}\n"
        f"Type:     {order_type}\n"
        f"Product:  {product}"
    )
    if amo:
        order_info += "\nVariety:  [yellow]AMO[/yellow] (After Market Order)"
    if price:
        order_info += f"\nPrice:    ₹{price:.2f}"
    if sl:
        order_info += f"\nSL:       ₹{sl:.2f}"
    if target:
        order_info += f"\nTarget:   ₹{target:.2f}"
    
    console.print(Panel(order_info, title="[bold cyan]Placing Order[/bold cyan]", border_style="cyan"))
    
    try:
        broker = _get_broker(config)
        result = broker.place_order(order)
        
        if result.status in ("COMPLETE", "OPEN", "PLACED"):
            # Success
            result_text = (
                f"[bold green]Order Executed Successfully[/bold green]\n\n"
                f"Order ID:     {result.order_id}\n"
                f"Status:       {result.status}"
            )
            if result.filled_qty > 0:
                result_text += f"\nFilled Qty:   {result.filled_qty}"
                result_text += f"\nFilled Price: ₹{result.filled_price:.2f}"
                result_text += f"\nValue:        ₹{result.filled_price * result.filled_qty:,.2f}"
            if result.message:
                result_text += f"\n\n[dim]{result.message}[/dim]"
            
            console.print(Panel(result_text, title="[bold green]Success[/bold green]", border_style="green"))
            
            # Log trade to journal
            _log_trade(
                symbol=symbol,
                side="BUY",
                quantity=qty,
                price=result.filled_price if result.filled_price > 0 else (price or 0),
                order_id=result.order_id,
                is_paper=config.get("trading", {}).get("mode", "paper") == "paper",
            )
            
            # Place SL order if specified
            if sl and result.status == "COMPLETE":
                _place_sl_order(config, symbol, qty, sl, product)
        else:
            # Failed
            console.print(Panel(
                f"[red]Order Failed[/red]\n\n"
                f"Status: {result.status}\n"
                f"Message: {result.message}",
                title="[bold red]Error[/bold red]",
                border_style="red",
            ))
            raise SystemExit(1)
            
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to place order:[/red]\n\n{str(e)}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)


def _place_sl_order(config: dict, symbol: str, qty: int, sl_price: float, product: str) -> None:
    """Place a stop-loss order after a buy executes.
    
    Args:
        config: Application configuration.
        symbol: Trading symbol.
        qty: Quantity.
        sl_price: Stop-loss trigger price.
        product: Product type (MIS/CNC).
    """
    from daytrader.models import Order
    
    sl_order = Order(
        symbol=symbol,
        side="SELL",
        quantity=qty,
        order_type="SL-M",
        product=product,
        price=None,
        trigger_price=sl_price,
    )
    
    console.print(f"\n[dim]Placing stop-loss order at ₹{sl_price:.2f}...[/dim]")
    
    try:
        broker = _get_broker(config)
        result = broker.place_order(sl_order)
        
        if result.status == "COMPLETE" or result.status == "OPEN" or result.status == "TRIGGER_PENDING":
            console.print(f"[green]✓ Stop-loss order placed: {result.order_id}[/green]")
        else:
            console.print(f"[yellow]⚠ Stop-loss order failed: {result.message}[/yellow]")
    except Exception as e:
        console.print(f"[yellow]⚠ Failed to place stop-loss: {str(e)}[/yellow]")


@click.command()
@click.argument("symbol")
@click.argument("qty", type=int)
@click.option(
    "-p", "--price",
    type=float,
    default=None,
    help="Limit price for LIMIT orders. If not specified, places a MARKET order.",
)
@click.option(
    "-d", "--delivery",
    is_flag=True,
    default=False,
    help="Use CNC (delivery) product type instead of MIS (intraday).",
)
@click.option(
    "-a", "--amo",
    is_flag=True,
    default=False,
    help="Place as After Market Order (executes at next market open).",
)
def sell(
    symbol: str,
    qty: int,
    price: Optional[float],
    delivery: bool,
    amo: bool,
) -> None:
    """Place a sell order for a symbol.
    
    SYMBOL is the trading symbol (e.g., RELIANCE, INFY, TCS).
    QTY is the number of shares to sell.
    
    By default, places a MARKET order with MIS (intraday) product type.
    Use --price for LIMIT orders and --delivery for CNC product type.
    Use --amo to place after market hours (executes at next market open).
    
    \b
    Examples:
      daytrader sell RELIANCE 10          # Market sell 10 shares
      daytrader sell INFY 5 --price 1600  # Limit sell at ₹1600
      daytrader sell TCS 10 --delivery    # Delivery (CNC) order
      daytrader sell TCS 1 --delivery --amo  # AMO delivery sell
    """
    from daytrader.models import Order
    
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
    product = "CNC" if delivery else "MIS"
    order_type = "LIMIT" if price is not None else "MARKET"
    variety = "AMO" if amo else "NORMAL"
    
    # Create the order
    order = Order(
        symbol=symbol,
        side="SELL",
        quantity=qty,
        order_type=order_type,
        product=product,
        variety=variety,
        price=price,
        trigger_price=None,
    )
    
    # Display order details before execution
    order_info = (
        f"[bold]Order Details[/bold]\n\n"
        f"Symbol:   {symbol}\n"
        f"Side:     [red]SELL[/red]\n"
        f"Quantity: {qty}\n"
        f"Type:     {order_type}\n"
        f"Product:  {product}"
    )
    if amo:
        order_info += "\nVariety:  [yellow]AMO[/yellow] (After Market Order)"
    if price:
        order_info += f"\nPrice:    ₹{price:.2f}"
    
    console.print(Panel(order_info, title="[bold cyan]Placing Order[/bold cyan]", border_style="cyan"))
    
    try:
        broker = _get_broker(config)
        result = broker.place_order(order)
        
        if result.status in ("COMPLETE", "OPEN", "PLACED"):
            # Success
            result_text = (
                f"[bold green]Order Executed Successfully[/bold green]\n\n"
                f"Order ID:     {result.order_id}\n"
                f"Status:       {result.status}"
            )
            if result.filled_qty > 0:
                result_text += f"\nFilled Qty:   {result.filled_qty}"
                result_text += f"\nFilled Price: ₹{result.filled_price:.2f}"
                result_text += f"\nValue:        ₹{result.filled_price * result.filled_qty:,.2f}"
            if result.message:
                result_text += f"\n\n[dim]{result.message}[/dim]"
            
            console.print(Panel(result_text, title="[bold green]Success[/bold green]", border_style="green"))
            
            # Log trade to journal
            _log_trade(
                symbol=symbol,
                side="SELL",
                quantity=qty,
                price=result.filled_price if result.filled_price > 0 else (price or 0),
                order_id=result.order_id,
                is_paper=config.get("trading", {}).get("mode", "paper") == "paper",
            )
        else:
            # Failed
            console.print(Panel(
                f"[red]Order Failed[/red]\n\n"
                f"Status: {result.status}\n"
                f"Message: {result.message}",
                title="[bold red]Error[/bold red]",
                border_style="red",
            ))
            raise SystemExit(1)
            
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to place order:[/red]\n\n{str(e)}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)


@click.command()
def positions() -> None:
    """Display all open positions with current P&L.
    
    Shows symbol, quantity, average price, LTP, P&L amount and percentage
    for all open positions.
    
    \b
    Examples:
      daytrader positions
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
        pos_list = broker.get_positions()
        
        if not pos_list:
            console.print(Panel(
                "[dim]No open positions[/dim]",
                title="[bold]Positions[/bold]",
                border_style="dim",
            ))
            return
        
        # Create positions table
        table = Table(
            title="Open Positions",
            show_header=True,
            header_style="bold cyan",
        )
        
        table.add_column("Symbol", style="bold")
        table.add_column("Qty", justify="right")
        table.add_column("Avg Price", justify="right")
        table.add_column("LTP", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")
        table.add_column("Product", justify="center", style="dim")
        
        total_pnl = 0.0
        
        for pos in pos_list:
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
            total_pnl += pos.pnl
        
        console.print(table)
        
        # Show total P&L
        total_color = "green" if total_pnl >= 0 else "red"
        total_sign = "+" if total_pnl >= 0 else ""
        console.print(f"\n[bold]Total P&L:[/bold] [{total_color}]{total_sign}₹{total_pnl:.2f}[/{total_color}]")
        
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to get positions:[/red]\n\n{str(e)}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)


@click.command("exit")
@click.argument("symbol", required=False)
@click.option(
    "--all",
    "exit_all",
    is_flag=True,
    default=False,
    help="Exit all open positions.",
)
def exit_position(symbol: Optional[str], exit_all: bool) -> None:
    """Exit (close) open positions.
    
    SYMBOL is the trading symbol to exit. Use --all to exit all positions.
    
    \b
    Examples:
      daytrader exit RELIANCE    # Exit RELIANCE position
      daytrader exit --all       # Exit all positions
    """
    from daytrader.models import Order
    
    config = _get_config()
    
    if config is None:
        console.print(Panel(
            "[red]Configuration not found.[/red]\n\n"
            "Run [cyan]daytrader login[/cyan] to create a config file.",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    if not symbol and not exit_all:
        console.print(Panel(
            "[red]Please specify a symbol or use --all to exit all positions.[/red]",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    try:
        broker = _get_broker(config)
        pos_list = broker.get_positions()
        
        if not pos_list:
            console.print(Panel(
                "[dim]No open positions to exit[/dim]",
                title="[bold]Exit[/bold]",
                border_style="dim",
            ))
            return
        
        # Filter positions to exit
        if exit_all:
            positions_to_exit = pos_list
        else:
            symbol = symbol.upper()
            positions_to_exit = [p for p in pos_list if p.symbol == symbol]
            
            if not positions_to_exit:
                console.print(Panel(
                    f"[yellow]No open position found for {symbol}[/yellow]",
                    title="[bold yellow]Not Found[/bold yellow]",
                    border_style="yellow",
                ))
                return
        
        # Exit each position
        success_count = 0
        fail_count = 0
        total_pnl = 0.0
        
        for pos in positions_to_exit:
            # Create sell order to close position
            order = Order(
                symbol=pos.symbol,
                side="SELL",
                quantity=pos.quantity,
                order_type="MARKET",
                product=pos.product,
                price=None,
                trigger_price=None,
            )
            
            console.print(f"[dim]Exiting {pos.symbol} ({pos.quantity} shares)...[/dim]")
            
            result = broker.place_order(order)
            
            if result.status == "COMPLETE" or result.status == "OPEN":
                pnl = (result.filled_price - pos.average_price) * pos.quantity
                total_pnl += pnl
                pnl_color = "green" if pnl >= 0 else "red"
                pnl_sign = "+" if pnl >= 0 else ""
                console.print(
                    f"[green]✓[/green] {pos.symbol}: Sold {result.filled_qty} @ ₹{result.filled_price:.2f} "
                    f"[{pnl_color}]({pnl_sign}₹{pnl:.2f})[/{pnl_color}]"
                )
                success_count += 1
            else:
                console.print(f"[red]✗[/red] {pos.symbol}: {result.message}")
                fail_count += 1
        
        # Summary
        console.print()
        if success_count > 0:
            total_color = "green" if total_pnl >= 0 else "red"
            total_sign = "+" if total_pnl >= 0 else ""
            console.print(Panel(
                f"[bold]Exited {success_count} position(s)[/bold]\n\n"
                f"Realized P&L: [{total_color}]{total_sign}₹{total_pnl:.2f}[/{total_color}]",
                title="[bold green]Exit Complete[/bold green]",
                border_style="green",
            ))
        
        if fail_count > 0:
            console.print(f"[yellow]⚠ {fail_count} position(s) failed to exit[/yellow]")
            
    except Exception as e:
        console.print(Panel(
            f"[red]Failed to exit positions:[/red]\n\n{str(e)}",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
