"""Database query tools for AI agents.

These tools allow agents to query the SQLite database for candles,
watchlist symbols, and positions.
"""

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from daytrader.db.store import DataStore
from daytrader.models import Candle, Position


# Default database path
DEFAULT_DB_PATH = Path.home() / ".config" / "daytrader" / "daytrader.db"


def _get_data_store(db_path: Optional[Path] = None) -> DataStore:
    """Get a DataStore instance.
    
    Args:
        db_path: Optional path to database. Uses default if not provided.
        
    Returns:
        DataStore instance.
    """
    return DataStore(db_path or DEFAULT_DB_PATH)


def query_candles(
    symbol: str,
    timeframe: str = "1day",
    days: int = 30,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    db_path: Optional[Path] = None,
) -> dict:
    """Query candle data from the database.
    
    This tool retrieves OHLCV (Open, High, Low, Close, Volume) data
    for a given symbol from the SQLite database.
    
    Args:
        symbol: Trading symbol (e.g., "RELIANCE", "TCS").
        timeframe: Candle timeframe - one of "1min", "5min", "15min", "1hour", "1day".
        days: Number of days of data to retrieve (default 30). Ignored if from_date is set.
        from_date: Start date for data retrieval. If None, uses (today - days).
        to_date: End date for data retrieval. If None, uses today.
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - symbol: The queried symbol
        - timeframe: The timeframe used
        - count: Number of candles returned
        - candles: List of candle dictionaries with timestamp, open, high, low, close, volume
        - latest: The most recent candle (if any)
        - error: Error message if query failed (None if successful)
    """
    try:
        store = _get_data_store(db_path)
        
        # Determine date range
        end_date = to_date or date.today()
        start_date = from_date or (end_date - timedelta(days=days))
        
        # Query candles
        candles = store.get_candles(
            symbol=symbol.upper(),
            timeframe=timeframe,
            from_date=start_date,
            to_date=end_date,
        )
        
        # Format candles for output
        candle_list = [
            {
                "timestamp": c.timestamp.isoformat(),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "volume": c.volume,
            }
            for c in candles
        ]
        
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "from_date": start_date.isoformat(),
            "to_date": end_date.isoformat(),
            "count": len(candle_list),
            "candles": candle_list,
            "latest": candle_list[-1] if candle_list else None,
            "error": None,
        }
        
    except Exception as e:
        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "count": 0,
            "candles": [],
            "latest": None,
            "error": str(e),
        }


def get_watchlist(
    list_name: str = "default",
    db_path: Optional[Path] = None,
) -> dict:
    """Get symbols from a watchlist.
    
    This tool retrieves all symbols in a named watchlist from the database.
    
    Args:
        list_name: Name of the watchlist (default "default").
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - list_name: The watchlist name
        - symbols: List of symbols in the watchlist
        - count: Number of symbols
        - error: Error message if query failed (None if successful)
    """
    try:
        store = _get_data_store(db_path)
        
        symbols = store.get_watchlist(list_name=list_name)
        
        return {
            "list_name": list_name,
            "symbols": symbols,
            "count": len(symbols),
            "error": None,
        }
        
    except Exception as e:
        return {
            "list_name": list_name,
            "symbols": [],
            "count": 0,
            "error": str(e),
        }


def get_positions(db_path: Optional[Path] = None) -> dict:
    """Get all open positions from the database.
    
    This tool retrieves all open trading positions stored in the database.
    Note: This returns cached positions. For live positions, use broker tools.
    
    Args:
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - positions: List of position dictionaries with symbol, quantity, 
                    average_price, ltp, pnl, pnl_percent, product
        - count: Number of positions
        - total_pnl: Sum of all position P&L
        - error: Error message if query failed (None if successful)
    """
    try:
        store = _get_data_store(db_path)
        
        positions = store.get_positions()
        
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


def get_trades(
    trade_date: Optional[date] = None,
    db_path: Optional[Path] = None,
) -> dict:
    """Get trades from the database.
    
    This tool retrieves trade history from the database.
    
    Args:
        trade_date: Optional date filter. If None, returns all trades.
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - trades: List of trade dictionaries
        - count: Number of trades
        - error: Error message if query failed (None if successful)
    """
    try:
        store = _get_data_store(db_path)
        
        trades = store.get_trades(trade_date=trade_date)
        
        trade_list = [
            {
                "id": t.id,
                "timestamp": t.timestamp.isoformat(),
                "symbol": t.symbol,
                "side": t.side,
                "quantity": t.quantity,
                "price": t.price,
                "order_id": t.order_id,
                "pnl": t.pnl,
                "is_paper": t.is_paper,
            }
            for t in trades
        ]
        
        return {
            "trades": trade_list,
            "count": len(trade_list),
            "date_filter": trade_date.isoformat() if trade_date else None,
            "error": None,
        }
        
    except Exception as e:
        return {
            "trades": [],
            "count": 0,
            "date_filter": trade_date.isoformat() if trade_date else None,
            "error": str(e),
        }


def get_alerts(db_path: Optional[Path] = None) -> dict:
    """Get all active alerts from the database.
    
    This tool retrieves all alerts stored in the database.
    
    Args:
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - alerts: List of alert dictionaries
        - count: Number of alerts
        - error: Error message if query failed (None if successful)
    """
    try:
        store = _get_data_store(db_path)
        
        alerts = store.get_alerts()
        
        alert_list = [
            {
                "id": a.id,
                "symbol": a.symbol,
                "condition": a.condition,
                "created_at": a.created_at.isoformat(),
                "triggered": a.triggered,
            }
            for a in alerts
        ]
        
        return {
            "alerts": alert_list,
            "count": len(alert_list),
            "active_count": sum(1 for a in alerts if not a.triggered),
            "error": None,
        }
        
    except Exception as e:
        return {
            "alerts": [],
            "count": 0,
            "active_count": 0,
            "error": str(e),
        }
