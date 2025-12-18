"""Broker tools for AI agents.

These tools allow agents to interact with the broker for placing orders,
getting positions, and checking account balance.
"""

from pathlib import Path
from typing import Literal, Optional

from daytrader.brokers.base import BaseBroker, Balance
from daytrader.brokers.paper import PaperBroker
from daytrader.db.store import DataStore
from daytrader.models import Order, OrderResult, Position


# Default database path
DEFAULT_DB_PATH = Path.home() / ".config" / "daytrader" / "daytrader.db"

# Global broker instance (set by application)
_broker: Optional[BaseBroker] = None


def set_broker(broker: BaseBroker) -> None:
    """Set the global broker instance.
    
    This should be called during application initialization to set
    the broker that will be used by the tools.
    
    Args:
        broker: Broker instance to use.
    """
    global _broker
    _broker = broker


def get_broker() -> Optional[BaseBroker]:
    """Get the current broker instance.
    
    Returns:
        Current broker instance or None if not set.
    """
    return _broker


def _get_default_broker(db_path: Optional[Path] = None) -> BaseBroker:
    """Get a default broker instance (paper trading).
    
    Args:
        db_path: Optional path to database file.
        
    Returns:
        PaperBroker instance.
    """
    store = DataStore(db_path or DEFAULT_DB_PATH)
    return PaperBroker(data_store=store)


def _ensure_broker(db_path: Optional[Path] = None) -> BaseBroker:
    """Ensure a broker is available.
    
    Args:
        db_path: Optional path to database file.
        
    Returns:
        Broker instance.
    """
    if _broker is not None:
        return _broker
    return _get_default_broker(db_path)


def place_order(
    symbol: str,
    side: Literal["BUY", "SELL"],
    quantity: int,
    order_type: Literal["MARKET", "LIMIT", "SL", "SL-M"] = "MARKET",
    product: Literal["MIS", "CNC"] = "MIS",
    price: Optional[float] = None,
    trigger_price: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> dict:
    """Place an order through the broker.
    
    This tool places buy or sell orders through the configured broker.
    In paper trading mode, orders are simulated.
    
    Args:
        symbol: Trading symbol (e.g., "RELIANCE", "TCS").
        side: Order side - "BUY" or "SELL".
        quantity: Number of shares to trade.
        order_type: Order type - "MARKET", "LIMIT", "SL", or "SL-M".
        product: Product type - "MIS" (intraday) or "CNC" (delivery).
        price: Limit price (required for LIMIT orders).
        trigger_price: Trigger price (required for SL orders).
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - success: Whether order was placed successfully
        - order_id: Order ID if successful
        - status: Order status
        - filled_qty: Quantity filled
        - filled_price: Average fill price
        - message: Status message
        - error: Error message if order failed (None if successful)
    """
    try:
        broker = _ensure_broker(db_path)
        
        # Validate inputs
        if quantity <= 0:
            return {
                "success": False,
                "order_id": None,
                "status": "REJECTED",
                "filled_qty": 0,
                "filled_price": 0.0,
                "message": "Quantity must be positive",
                "error": "Invalid quantity",
            }
        
        if order_type == "LIMIT" and price is None:
            return {
                "success": False,
                "order_id": None,
                "status": "REJECTED",
                "filled_qty": 0,
                "filled_price": 0.0,
                "message": "Price required for LIMIT orders",
                "error": "Missing price",
            }
        
        if order_type in ("SL", "SL-M") and trigger_price is None:
            return {
                "success": False,
                "order_id": None,
                "status": "REJECTED",
                "filled_qty": 0,
                "filled_price": 0.0,
                "message": "Trigger price required for SL orders",
                "error": "Missing trigger price",
            }
        
        # Create order
        order = Order(
            symbol=symbol.upper(),
            side=side,
            quantity=quantity,
            order_type=order_type,
            product=product,
            price=price,
            trigger_price=trigger_price,
        )
        
        # Place order
        result = broker.place_order(order)
        
        success = result.status in ("COMPLETE", "OPEN", "PENDING")
        
        return {
            "success": success,
            "order_id": result.order_id,
            "status": result.status,
            "filled_qty": result.filled_qty,
            "filled_price": result.filled_price,
            "message": result.message,
            "error": None if success else result.message,
        }
        
    except Exception as e:
        return {
            "success": False,
            "order_id": None,
            "status": "ERROR",
            "filled_qty": 0,
            "filled_price": 0.0,
            "message": str(e),
            "error": str(e),
        }


def get_positions(db_path: Optional[Path] = None) -> dict:
    """Get all open positions from the broker.
    
    This tool retrieves current open positions from the broker.
    
    Args:
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - positions: List of position dictionaries
        - count: Number of positions
        - total_pnl: Total P&L across all positions
        - error: Error message if query failed (None if successful)
    """
    try:
        broker = _ensure_broker(db_path)
        
        positions = broker.get_positions()
        
        position_list = [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "average_price": p.average_price,
                "ltp": p.ltp,
                "pnl": p.pnl,
                "pnl_percent": p.pnl_percent,
                "product": p.product,
            }
            for p in positions
        ]
        
        total_pnl = sum(p.pnl for p in positions)
        
        return {
            "positions": position_list,
            "count": len(position_list),
            "total_pnl": total_pnl,
            "error": None,
        }
        
    except Exception as e:
        return {
            "positions": [],
            "count": 0,
            "total_pnl": 0.0,
            "error": str(e),
        }


def get_balance(db_path: Optional[Path] = None) -> dict:
    """Get account balance from the broker.
    
    This tool retrieves current account balance and margin information.
    
    Args:
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - available_cash: Available cash for trading
        - used_margin: Margin currently in use
        - total_value: Total portfolio value
        - error: Error message if query failed (None if successful)
    """
    try:
        broker = _ensure_broker(db_path)
        
        balance = broker.get_balance()
        
        return {
            "available_cash": balance.available_cash,
            "used_margin": balance.used_margin,
            "total_value": balance.total_value,
            "error": None,
        }
        
    except Exception as e:
        return {
            "available_cash": 0.0,
            "used_margin": 0.0,
            "total_value": 0.0,
            "error": str(e),
        }


def get_quote(
    symbol: str,
    db_path: Optional[Path] = None,
) -> dict:
    """Get real-time quote for a symbol.
    
    This tool retrieves the current market quote for a symbol.
    
    Args:
        symbol: Trading symbol (e.g., "RELIANCE", "TCS").
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - symbol: The queried symbol
        - ltp: Last traded price
        - change: Price change
        - change_percent: Percentage change
        - volume: Trading volume
        - open: Opening price
        - high: Day high
        - low: Day low
        - close: Previous close
        - error: Error message if query failed (None if successful)
    """
    try:
        broker = _ensure_broker(db_path)
        
        quote = broker.get_quote(symbol.upper())
        
        return {
            "symbol": quote.symbol,
            "ltp": quote.ltp,
            "change": quote.change,
            "change_percent": quote.change_percent,
            "volume": quote.volume,
            "open": quote.open,
            "high": quote.high,
            "low": quote.low,
            "close": quote.close,
            "error": None,
        }
        
    except Exception as e:
        return {
            "symbol": symbol.upper(),
            "ltp": 0.0,
            "change": 0.0,
            "change_percent": 0.0,
            "volume": 0,
            "open": 0.0,
            "high": 0.0,
            "low": 0.0,
            "close": 0.0,
            "error": str(e),
        }


def cancel_order(
    order_id: str,
    db_path: Optional[Path] = None,
) -> dict:
    """Cancel an open order.
    
    This tool cancels an open order by its ID.
    
    Args:
        order_id: Order ID to cancel.
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - success: Whether cancellation was successful
        - order_id: The order ID
        - message: Status message
        - error: Error message if cancellation failed (None if successful)
    """
    try:
        broker = _ensure_broker(db_path)
        
        success = broker.cancel_order(order_id)
        
        if success:
            return {
                "success": True,
                "order_id": order_id,
                "message": "Order cancelled successfully",
                "error": None,
            }
        else:
            return {
                "success": False,
                "order_id": order_id,
                "message": "Failed to cancel order",
                "error": "Order may have already been executed or cancelled",
            }
        
    except Exception as e:
        return {
            "success": False,
            "order_id": order_id,
            "message": str(e),
            "error": str(e),
        }


def exit_position(
    symbol: str,
    db_path: Optional[Path] = None,
) -> dict:
    """Exit an entire position for a symbol.
    
    This tool closes the entire position for a given symbol by placing
    a market order in the opposite direction.
    
    Args:
        symbol: Trading symbol to exit.
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - success: Whether exit was successful
        - symbol: The symbol
        - quantity: Quantity exited
        - order_id: Order ID if successful
        - message: Status message
        - error: Error message if exit failed (None if successful)
    """
    try:
        broker = _ensure_broker(db_path)
        
        # Get current position
        positions = broker.get_positions()
        position = next(
            (p for p in positions if p.symbol.upper() == symbol.upper()),
            None
        )
        
        if position is None:
            return {
                "success": False,
                "symbol": symbol.upper(),
                "quantity": 0,
                "order_id": None,
                "message": f"No open position for {symbol.upper()}",
                "error": "Position not found",
            }
        
        # Determine exit side
        exit_side = "SELL" if position.quantity > 0 else "BUY"
        exit_qty = abs(position.quantity)
        
        # Place exit order
        result = place_order(
            symbol=symbol,
            side=exit_side,
            quantity=exit_qty,
            order_type="MARKET",
            product=position.product,
            db_path=db_path,
        )
        
        if result["success"]:
            return {
                "success": True,
                "symbol": symbol.upper(),
                "quantity": exit_qty,
                "order_id": result["order_id"],
                "message": f"Exited {exit_qty} shares of {symbol.upper()}",
                "error": None,
            }
        else:
            return {
                "success": False,
                "symbol": symbol.upper(),
                "quantity": 0,
                "order_id": None,
                "message": result["message"],
                "error": result["error"],
            }
        
    except Exception as e:
        return {
            "success": False,
            "symbol": symbol.upper(),
            "quantity": 0,
            "order_id": None,
            "message": str(e),
            "error": str(e),
        }


def exit_all_positions(db_path: Optional[Path] = None) -> dict:
    """Exit all open positions.
    
    This tool closes all open positions by placing market orders.
    
    Args:
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - success: Whether all exits were successful
        - exited: List of successfully exited positions
        - failed: List of failed exits
        - total_exited: Number of positions exited
        - error: Error message if any exits failed (None if all successful)
    """
    try:
        broker = _ensure_broker(db_path)
        
        positions = broker.get_positions()
        
        if not positions:
            return {
                "success": True,
                "exited": [],
                "failed": [],
                "total_exited": 0,
                "error": None,
            }
        
        exited = []
        failed = []
        
        for position in positions:
            result = exit_position(position.symbol, db_path=db_path)
            
            if result["success"]:
                exited.append({
                    "symbol": position.symbol,
                    "quantity": abs(position.quantity),
                    "order_id": result["order_id"],
                })
            else:
                failed.append({
                    "symbol": position.symbol,
                    "error": result["error"],
                })
        
        return {
            "success": len(failed) == 0,
            "exited": exited,
            "failed": failed,
            "total_exited": len(exited),
            "error": f"Failed to exit {len(failed)} positions" if failed else None,
        }
        
    except Exception as e:
        return {
            "success": False,
            "exited": [],
            "failed": [],
            "total_exited": 0,
            "error": str(e),
        }
