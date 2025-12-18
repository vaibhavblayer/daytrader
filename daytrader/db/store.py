"""SQLite data store for DayTrader."""

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from daytrader.models import (
    Alert,
    Candle,
    JournalEntry,
    Position,
    Trade,
)


class DataStore:
    """SQLite-based data store for DayTrader."""

    REQUIRED_TABLES = [
        "candles",
        "trades",
        "positions",
        "watchlist",
        "alerts",
        "research_cache",
        "journal",
    ]

    def __init__(self, db_path: Path):
        """Initialize the data store.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_schema()

    def _ensure_db_dir(self) -> None:
        """Ensure the database directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


    def _init_schema(self) -> None:
        """Initialize database schema on first run."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Candles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)

            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    order_id TEXT NOT NULL,
                    pnl REAL,
                    is_paper INTEGER NOT NULL DEFAULT 0
                )
            """)

            # Positions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL UNIQUE,
                    quantity INTEGER NOT NULL,
                    average_price REAL NOT NULL,
                    ltp REAL NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_percent REAL NOT NULL,
                    product TEXT NOT NULL
                )
            """)

            # Watchlist table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    list_name TEXT NOT NULL DEFAULT 'default',
                    UNIQUE(symbol, list_name)
                )
            """)

            # Alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    triggered INTEGER NOT NULL DEFAULT 0
                )
            """)

            # Research cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS research_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    cached_at TEXT NOT NULL
                )
            """)

            # Journal table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    trades_count INTEGER NOT NULL,
                    total_pnl REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    notes TEXT,
                    ai_insights TEXT
                )
            """)

            conn.commit()
        finally:
            conn.close()


    def get_tables(self) -> list[str]:
        """Get list of all tables in the database."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            return [row["name"] for row in cursor.fetchall()]
        finally:
            conn.close()

    # ==================== Candles ====================

    def save_candles(
        self, symbol: str, timeframe: str, candles: list[Candle]
    ) -> None:
        """Save candles to the database.

        Args:
            symbol: Trading symbol.
            timeframe: Candle timeframe (e.g., '1day', '5min').
            candles: List of candles to save.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            for candle in candles:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO candles 
                    (symbol, timeframe, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        symbol,
                        timeframe,
                        candle.timestamp.isoformat(),
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.volume,
                    ),
                )
            conn.commit()
        finally:
            conn.close()

    def get_candles(
        self,
        symbol: str,
        timeframe: str,
        from_date: date,
        to_date: date,
    ) -> list[Candle]:
        """Get candles from the database.

        Args:
            symbol: Trading symbol.
            timeframe: Candle timeframe.
            from_date: Start date.
            to_date: End date.

        Returns:
            List of candles in the date range.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, open, high, low, close, volume
                FROM candles
                WHERE symbol = ? AND timeframe = ?
                AND date(timestamp) >= ? AND date(timestamp) <= ?
                ORDER BY timestamp
                """,
                (symbol, timeframe, from_date.isoformat(), to_date.isoformat()),
            )
            return [
                Candle(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()


    # ==================== Trades ====================

    def log_trade(self, trade: Trade) -> None:
        """Log a trade to the database.

        Args:
            trade: Trade to log.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO trades 
                (timestamp, symbol, side, quantity, price, order_id, pnl, is_paper)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.timestamp.isoformat(),
                    trade.symbol,
                    trade.side,
                    trade.quantity,
                    trade.price,
                    trade.order_id,
                    trade.pnl,
                    1 if trade.is_paper else 0,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_trades(self, trade_date: Optional[date] = None) -> list[Trade]:
        """Get trades from the database.

        Args:
            trade_date: Optional date filter. If None, returns all trades.

        Returns:
            List of trades.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if trade_date:
                cursor.execute(
                    """
                    SELECT id, timestamp, symbol, side, quantity, price, order_id, pnl, is_paper
                    FROM trades
                    WHERE date(timestamp) = ?
                    ORDER BY timestamp
                    """,
                    (trade_date.isoformat(),),
                )
            else:
                cursor.execute(
                    """
                    SELECT id, timestamp, symbol, side, quantity, price, order_id, pnl, is_paper
                    FROM trades
                    ORDER BY timestamp
                    """
                )
            return [
                Trade(
                    id=row["id"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    symbol=row["symbol"],
                    side=row["side"],
                    quantity=row["quantity"],
                    price=row["price"],
                    order_id=row["order_id"],
                    pnl=row["pnl"],
                    is_paper=bool(row["is_paper"]),
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    # ==================== Positions ====================

    def save_position(self, position: Position) -> None:
        """Save or update a position.

        Args:
            position: Position to save.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO positions 
                (symbol, quantity, average_price, ltp, pnl, pnl_percent, product)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position.symbol,
                    position.quantity,
                    position.average_price,
                    position.ltp,
                    position.pnl,
                    position.pnl_percent,
                    position.product,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_positions(self) -> list[Position]:
        """Get all open positions.

        Returns:
            List of positions.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT symbol, quantity, average_price, ltp, pnl, pnl_percent, product
                FROM positions
                """
            )
            return [
                Position(
                    symbol=row["symbol"],
                    quantity=row["quantity"],
                    average_price=row["average_price"],
                    ltp=row["ltp"],
                    pnl=row["pnl"],
                    pnl_percent=row["pnl_percent"],
                    product=row["product"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def delete_position(self, symbol: str) -> None:
        """Delete a position.

        Args:
            symbol: Symbol to delete.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            conn.commit()
        finally:
            conn.close()


    # ==================== Watchlist ====================

    def add_to_watchlist(self, symbol: str, list_name: str = "default") -> None:
        """Add a symbol to a watchlist.

        Args:
            symbol: Symbol to add.
            list_name: Name of the watchlist.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO watchlist (symbol, list_name)
                VALUES (?, ?)
                """,
                (symbol, list_name),
            )
            conn.commit()
        finally:
            conn.close()

    def remove_from_watchlist(self, symbol: str, list_name: str = "default") -> None:
        """Remove a symbol from a watchlist.

        Args:
            symbol: Symbol to remove.
            list_name: Name of the watchlist.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM watchlist WHERE symbol = ? AND list_name = ?",
                (symbol, list_name),
            )
            conn.commit()
        finally:
            conn.close()

    def get_watchlist(self, list_name: str = "default") -> list[str]:
        """Get all symbols in a watchlist.

        Args:
            list_name: Name of the watchlist.

        Returns:
            List of symbols.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT symbol FROM watchlist WHERE list_name = ?",
                (list_name,),
            )
            return [row["symbol"] for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_watchlist_names(self) -> list[str]:
        """Get all watchlist names.

        Returns:
            List of watchlist names.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT list_name FROM watchlist")
            return [row["list_name"] for row in cursor.fetchall()]
        finally:
            conn.close()

    # ==================== Alerts ====================

    def save_alert(self, alert: Alert) -> int:
        """Save an alert to the database.

        Args:
            alert: Alert to save.

        Returns:
            The ID of the saved alert.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO alerts (symbol, condition, created_at, triggered)
                VALUES (?, ?, ?, ?)
                """,
                (
                    alert.symbol,
                    alert.condition,
                    alert.created_at.isoformat(),
                    1 if alert.triggered else 0,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0
        finally:
            conn.close()

    def get_alerts(self) -> list[Alert]:
        """Get all alerts.

        Returns:
            List of alerts.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, symbol, condition, created_at, triggered
                FROM alerts
                ORDER BY created_at DESC
                """
            )
            return [
                Alert(
                    id=row["id"],
                    symbol=row["symbol"],
                    condition=row["condition"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    triggered=bool(row["triggered"]),
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    def get_alert_by_id(self, alert_id: int) -> Optional[Alert]:
        """Get an alert by ID.

        Args:
            alert_id: Alert ID.

        Returns:
            Alert if found, None otherwise.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, symbol, condition, created_at, triggered
                FROM alerts
                WHERE id = ?
                """,
                (alert_id,),
            )
            row = cursor.fetchone()
            if row:
                return Alert(
                    id=row["id"],
                    symbol=row["symbol"],
                    condition=row["condition"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    triggered=bool(row["triggered"]),
                )
            return None
        finally:
            conn.close()

    def delete_alert(self, alert_id: int) -> None:
        """Delete an alert.

        Args:
            alert_id: ID of the alert to delete.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
            conn.commit()
        finally:
            conn.close()

    def update_alert_triggered(self, alert_id: int, triggered: bool) -> None:
        """Update the triggered status of an alert.

        Args:
            alert_id: Alert ID.
            triggered: New triggered status.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE alerts SET triggered = ? WHERE id = ?",
                (1 if triggered else 0, alert_id),
            )
            conn.commit()
        finally:
            conn.close()


    # ==================== Research Cache ====================

    def cache_research(self, symbol: str, content: str, source: str) -> None:
        """Cache research results.

        Args:
            symbol: Trading symbol.
            content: Research content.
            source: Source of the research.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO research_cache (symbol, content, source, cached_at)
                VALUES (?, ?, ?, ?)
                """,
                (symbol, content, source, datetime.now().isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def get_cached_research(
        self, symbol: str, max_age_hours: int = 24
    ) -> Optional[str]:
        """Get cached research if not expired.

        Args:
            symbol: Trading symbol.
            max_age_hours: Maximum age of cache in hours.

        Returns:
            Cached content if found and not expired, None otherwise.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT content, cached_at
                FROM research_cache
                WHERE symbol = ?
                ORDER BY cached_at DESC
                LIMIT 1
                """,
                (symbol,),
            )
            row = cursor.fetchone()
            if row:
                cached_at = datetime.fromisoformat(row["cached_at"])
                age_hours = (datetime.now() - cached_at).total_seconds() / 3600
                if age_hours <= max_age_hours:
                    return row["content"]
            return None
        finally:
            conn.close()

    # ==================== Journal ====================

    def save_journal_entry(self, entry: JournalEntry) -> None:
        """Save a journal entry.

        Args:
            entry: Journal entry to save.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO journal 
                (date, trades_count, total_pnl, win_rate, notes, ai_insights)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.date.isoformat(),
                    entry.trades_count,
                    entry.total_pnl,
                    entry.win_rate,
                    entry.notes,
                    entry.ai_insights,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_journal(self, from_date: Optional[date] = None) -> list[JournalEntry]:
        """Get journal entries.

        Args:
            from_date: Optional start date filter.

        Returns:
            List of journal entries.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if from_date:
                cursor.execute(
                    """
                    SELECT date, trades_count, total_pnl, win_rate, notes, ai_insights
                    FROM journal
                    WHERE date >= ?
                    ORDER BY date DESC
                    """,
                    (from_date.isoformat(),),
                )
            else:
                cursor.execute(
                    """
                    SELECT date, trades_count, total_pnl, win_rate, notes, ai_insights
                    FROM journal
                    ORDER BY date DESC
                    """
                )
            return [
                JournalEntry(
                    date=date.fromisoformat(row["date"]),
                    trades_count=row["trades_count"],
                    total_pnl=row["total_pnl"],
                    win_rate=row["win_rate"],
                    notes=row["notes"],
                    ai_insights=row["ai_insights"],
                )
                for row in cursor.fetchall()
            ]
        finally:
            conn.close()

    # ==================== Stats ====================

    def get_stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dictionary with table record counts.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            stats = {}
            for table in self.REQUIRED_TABLES:
                cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                stats[table] = cursor.fetchone()["count"]
            return stats
        finally:
            conn.close()
