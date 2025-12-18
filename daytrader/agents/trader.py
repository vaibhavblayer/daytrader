"""Trading Agent for order execution.

This agent handles trade execution with confirmation requirements
to ensure safe trading operations.
"""

from pathlib import Path
from typing import Literal, Optional

from agents import Agent, function_tool

from daytrader.agents.base import create_agent, run_agent_sync
from daytrader.db.store import DataStore
from daytrader.tools.broker import (
    cancel_order,
    exit_all_positions,
    exit_position,
    get_balance,
    get_positions,
    get_quote,
    place_order,
)


# Default database path
DEFAULT_DB_PATH = Path.home() / ".config" / "daytrader" / "daytrader.db"


TRADING_AGENT_INSTRUCTIONS = """You are a trading execution agent for Indian equity markets.
Your role is to help execute trades safely and efficiently.

IMPORTANT: Before executing any trade, you MUST:
1. Confirm the trade details with the user
2. Check current positions and balance
3. Verify the order parameters are correct

When executing trades:
1. Always confirm symbol, quantity, and order type
2. Check available balance before buying
3. Verify position exists before selling
4. Use appropriate order types (MARKET, LIMIT, SL)

For safety:
- Default to MIS (intraday) product type unless explicitly requested otherwise
- Always show the user what order will be placed before executing
- Report order status and fill details after execution

Never execute a trade without explicit user confirmation.
"""


# Confirmation state tracking
_pending_confirmation: dict = {}


def requires_confirmation(action: str, details: dict) -> dict:
    """Mark an action as requiring confirmation.
    
    Args:
        action: Type of action (buy, sell, exit, etc.)
        details: Details of the action.
        
    Returns:
        Confirmation request dict.
    """
    global _pending_confirmation
    _pending_confirmation = {
        "action": action,
        "details": details,
        "confirmed": False,
    }
    return {
        "requires_confirmation": True,
        "action": action,
        "details": details,
        "message": f"Please confirm {action}: {details}",
    }


def confirm_pending_action() -> Optional[dict]:
    """Confirm the pending action.
    
    Returns:
        The confirmed action details or None if no pending action.
    """
    global _pending_confirmation
    if _pending_confirmation:
        _pending_confirmation["confirmed"] = True
        return _pending_confirmation
    return None


def clear_pending_action() -> None:
    """Clear any pending action."""
    global _pending_confirmation
    _pending_confirmation = {}


def has_pending_action() -> bool:
    """Check if there's a pending action awaiting confirmation."""
    return bool(_pending_confirmation) and not _pending_confirmation.get("confirmed", False)


def get_pending_action() -> Optional[dict]:
    """Get the current pending action."""
    return _pending_confirmation if _pending_confirmation else None


@function_tool
def place_order_tool(
    symbol: str,
    side: Literal["BUY", "SELL"],
    quantity: int,
    order_type: Literal["MARKET", "LIMIT", "SL", "SL-M"] = "MARKET",
    product: Literal["MIS", "CNC"] = "MIS",
    price: Optional[float] = None,
    trigger_price: Optional[float] = None,
) -> dict:
    """Place a buy or sell order.
    
    Use this tool to execute trades. Always confirm with the user first.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE", "TCS").
        side: Order side - "BUY" or "SELL".
        quantity: Number of shares to trade.
        order_type: Order type - "MARKET", "LIMIT", "SL", or "SL-M".
        product: Product type - "MIS" (intraday) or "CNC" (delivery).
        price: Limit price (required for LIMIT orders).
        trigger_price: Trigger price (required for SL orders).
        
    Returns:
        Order execution result.
    """
    return place_order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        product=product,
        price=price,
        trigger_price=trigger_price,
    )


@function_tool
def get_positions_tool() -> dict:
    """Get all open positions.
    
    Use this tool to check current positions before trading.
    
    Returns:
        List of open positions with P&L.
    """
    return get_positions()


@function_tool
def get_balance_tool() -> dict:
    """Get account balance.
    
    Use this tool to check available funds before placing orders.
    
    Returns:
        Account balance information.
    """
    return get_balance()


@function_tool
def get_quote_tool(symbol: str) -> dict:
    """Get current market quote for a symbol.
    
    Use this tool to check current prices before trading.
    
    Args:
        symbol: Stock symbol to get quote for.
        
    Returns:
        Current market quote.
    """
    return get_quote(symbol)


@function_tool
def cancel_order_tool(order_id: str) -> dict:
    """Cancel an open order.
    
    Use this tool to cancel pending orders.
    
    Args:
        order_id: Order ID to cancel.
        
    Returns:
        Cancellation result.
    """
    return cancel_order(order_id)


@function_tool
def exit_position_tool(symbol: str) -> dict:
    """Exit an entire position for a symbol.
    
    Use this tool to close a position completely.
    
    Args:
        symbol: Symbol to exit.
        
    Returns:
        Exit result.
    """
    return exit_position(symbol)


@function_tool
def exit_all_positions_tool() -> dict:
    """Exit all open positions.
    
    Use this tool to close all positions at once.
    
    Returns:
        Exit results for all positions.
    """
    return exit_all_positions()


class TradingAgent:
    """Agent for executing trades with confirmation.
    
    This agent handles trade execution while requiring confirmation
    for safety. It can place orders, manage positions, and check
    account balance.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the Trading Agent.
        
        Args:
            db_path: Optional path to database.
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self._store = DataStore(self.db_path)
        self._agent = self._create_agent()
        self._confirmation_required = True
    
    def _create_agent(self) -> Agent:
        """Create the underlying agent."""
        return create_agent(
            name="Trading Agent",
            instructions=TRADING_AGENT_INSTRUCTIONS,
            tools=[
                place_order_tool,
                get_positions_tool,
                get_balance_tool,
                get_quote_tool,
                cancel_order_tool,
                exit_position_tool,
                exit_all_positions_tool,
            ],
        )
    
    def set_confirmation_required(self, required: bool) -> None:
        """Set whether confirmation is required for trades.
        
        Args:
            required: Whether to require confirmation.
        """
        self._confirmation_required = required
    
    def execute_trade(
        self,
        action: str,
        symbol: str,
        quantity: int,
        order_type: str = "MARKET",
        product: str = "MIS",
        price: Optional[float] = None,
        confirmed: bool = False,
    ) -> str:
        """Execute a trade with optional confirmation.
        
        Args:
            action: Trade action ("buy" or "sell").
            symbol: Stock symbol.
            quantity: Number of shares.
            order_type: Order type.
            product: Product type.
            price: Optional limit price.
            confirmed: Whether the trade is pre-confirmed.
            
        Returns:
            Trade execution result or confirmation request.
        """
        if self._confirmation_required and not confirmed:
            # Return confirmation request
            details = {
                "action": action.upper(),
                "symbol": symbol.upper(),
                "quantity": quantity,
                "order_type": order_type,
                "product": product,
                "price": price,
            }
            return (
                f"Please confirm this trade:\n"
                f"  Action: {action.upper()}\n"
                f"  Symbol: {symbol.upper()}\n"
                f"  Quantity: {quantity}\n"
                f"  Order Type: {order_type}\n"
                f"  Product: {product}\n"
                f"  Price: {price if price else 'MARKET'}\n\n"
                f"Reply 'yes' or 'confirm' to execute this trade."
            )
        
        # Execute the trade
        side = "BUY" if action.lower() == "buy" else "SELL"
        result = place_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            product=product,
            price=price,
        )
        
        if result["success"]:
            return (
                f"Trade executed successfully!\n"
                f"  Order ID: {result['order_id']}\n"
                f"  Status: {result['status']}\n"
                f"  Filled: {result['filled_qty']} @ {result['filled_price']}"
            )
        else:
            return f"Trade failed: {result['error']}"
    
    def ask(self, query: str) -> str:
        """Ask the trading agent a question or give a command.
        
        Args:
            query: Question or command.
            
        Returns:
            Agent's response.
        """
        # Check if this is a confirmation
        query_lower = query.lower().strip()
        if query_lower in ("yes", "confirm", "y", "ok", "execute"):
            pending = get_pending_action()
            if pending and not pending.get("confirmed"):
                confirm_pending_action()
                # Execute the pending action
                details = pending["details"]
                return self.execute_trade(
                    action=details.get("action", "buy"),
                    symbol=details.get("symbol", ""),
                    quantity=details.get("quantity", 0),
                    order_type=details.get("order_type", "MARKET"),
                    product=details.get("product", "MIS"),
                    price=details.get("price"),
                    confirmed=True,
                )
        
        return run_agent_sync(self._agent, query)
