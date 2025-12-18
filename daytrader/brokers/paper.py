"""Paper trading broker implementation for simulated trading."""

import random
import uuid
from datetime import date, datetime
from typing import Optional

from daytrader.brokers.base import Balance, BaseBroker, Quote
from daytrader.db.store import DataStore
from daytrader.models import Candle, Order, OrderResult, Position, Trade


class PaperBroker(BaseBroker):
    """Paper trading broker for simulated trading.
    
    Simulates order execution with virtual balance, tracks positions
    in SQLite, and implements slippage simulation for realistic testing.
    """

    # Default slippage percentage for market orders
    DEFAULT_SLIPPAGE_PERCENT = 0.05  # 0.05%

    def __init__(
        self,
        data_store: DataStore,
        starting_balance: float = 100000.0,
        slippage_percent: float = DEFAULT_SLIPPAGE_PERCENT,
    ):
        """Initialize paper trading broker.
        
        Args:
            data_store: DataStore instance for persistence.
            starting_balance: Initial virtual balance.
            slippage_percent: Slippage percentage for market orders.
        """
        self._data_store = data_store
        self._starting_balance = starting_balance
        self._slippage_percent = slippage_percent
        self._authenticated = False
        
        # Initialize paper trading state
        self._init_paper_state()

    def _init_paper_state(self) -> None:
        """Initialize or load paper trading state from database."""
        # Paper trading state is stored in the database
        # We use a special "paper_state" entry in research_cache for balance
        cached = self._data_store.get_cached_research("__paper_balance__", max_age_hours=999999)
        if cached is None:
            self._balance = self._starting_balance
            self._save_balance()
        else:
            try:
                self._balance = float(cached)
            except ValueError:
                self._balance = self._starting_balance
                self._save_balance()

    def _save_balance(self) -> None:
        """Save current balance to database."""
        self._data_store.cache_research(
            symbol="__paper_balance__",
            content=str(self._balance),
            source="paper_broker",
        )

    def _get_simulated_price(self, symbol: str) -> float:
        """Get a simulated price for a symbol.
        
        In a real implementation, this would fetch from a data source.
        For paper trading, we use cached candle data or a default.
        
        Args:
            symbol: Trading symbol.
            
        Returns:
            Simulated current price.
        """
        # Try to get latest candle from database
        today = date.today()
        candles = self._data_store.get_candles(
            symbol=symbol,
            timeframe="1day",
            from_date=today,
            to_date=today,
        )
        
        if candles:
            return candles[-1].close
        
        # Fallback: check if we have any recent data
        from datetime import timedelta
        week_ago = today - timedelta(days=7)
        candles = self._data_store.get_candles(
            symbol=symbol,
            timeframe="1day",
            from_date=week_ago,
            to_date=today,
        )
        
        if candles:
            return candles[-1].close
        
        # Default price for testing
        return 100.0

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage to a price.
        
        Args:
            price: Base price.
            side: Order side (BUY or SELL).
            
        Returns:
            Price with slippage applied.
        """
        # Random slippage between 0 and max slippage
        slippage_factor = random.uniform(0, self._slippage_percent) / 100
        
        if side == "BUY":
            # Buyer pays more
            return price * (1 + slippage_factor)
        else:
            # Seller receives less
            return price * (1 - slippage_factor)

    def login(self) -> bool:
        """Authenticate (always succeeds for paper trading).
        
        Returns:
            True always.
        """
        self._authenticated = True
        return True

    def logout(self) -> bool:
        """Logout (always succeeds for paper trading).
        
        Returns:
            True always.
        """
        self._authenticated = False
        return True

    def is_authenticated(self) -> bool:
        """Check if authenticated.
        
        Returns:
            True if authenticated.
        """
        return self._authenticated

    def get_quote(self, symbol: str) -> Quote:
        """Get simulated quote for a symbol.
        
        Args:
            symbol: Trading symbol.
            
        Returns:
            Simulated quote.
        """
        price = self._get_simulated_price(symbol)
        
        # Simulate some variation
        variation = random.uniform(-0.02, 0.02)
        open_price = price * (1 + variation)
        high = max(price, open_price) * (1 + random.uniform(0, 0.01))
        low = min(price, open_price) * (1 - random.uniform(0, 0.01))
        prev_close = price * (1 - random.uniform(-0.02, 0.02))
        
        change = price - prev_close
        change_percent = (change / prev_close * 100) if prev_close > 0 else 0
        
        return Quote(
            symbol=symbol,
            ltp=price,
            change=change,
            change_percent=change_percent,
            volume=random.randint(10000, 1000000),
            open=open_price,
            high=high,
            low=low,
            close=prev_close,
        )

    def get_historical(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        interval: str,
    ) -> list[Candle]:
        """Get historical data from cache.
        
        Args:
            symbol: Trading symbol.
            from_date: Start date.
            to_date: End date.
            interval: Candle interval.
            
        Returns:
            List of candles from cache.
        """
        return self._data_store.get_candles(
            symbol=symbol,
            timeframe=interval,
            from_date=from_date,
            to_date=to_date,
        )

    def place_order(self, order: Order) -> OrderResult:
        """Place a simulated order.
        
        Args:
            order: Order to place.
            
        Returns:
            OrderResult with simulated execution.
        """
        # Get current price
        base_price = self._get_simulated_price(order.symbol)
        
        # Determine execution price
        if order.order_type == "MARKET":
            exec_price = self._apply_slippage(base_price, order.side)
        elif order.order_type == "LIMIT":
            if order.price is None:
                return OrderResult(
                    order_id="",
                    status="REJECTED",
                    filled_qty=0,
                    filled_price=0.0,
                    message="Limit price required for LIMIT orders",
                )
            # Check if limit price is favorable
            if order.side == "BUY" and order.price < base_price:
                # Limit buy below market - would need to wait
                exec_price = order.price
            elif order.side == "SELL" and order.price > base_price:
                # Limit sell above market - would need to wait
                exec_price = order.price
            else:
                # Execute at limit price
                exec_price = order.price
        else:
            # SL and SL-M orders
            if order.trigger_price is None:
                return OrderResult(
                    order_id="",
                    status="REJECTED",
                    filled_qty=0,
                    filled_price=0.0,
                    message="Trigger price required for SL orders",
                )
            exec_price = self._apply_slippage(order.trigger_price, order.side)
        
        # Calculate order value
        order_value = exec_price * order.quantity
        
        # Check balance for buy orders
        if order.side == "BUY":
            if order_value > self._balance:
                return OrderResult(
                    order_id="",
                    status="REJECTED",
                    filled_qty=0,
                    filled_price=0.0,
                    message=f"Insufficient balance. Required: {order_value:.2f}, Available: {self._balance:.2f}",
                )
            # Deduct from balance
            self._balance -= order_value
        else:
            # For sell, add to balance
            self._balance += order_value
        
        self._save_balance()
        
        # Generate order ID
        order_id = f"PAPER_{uuid.uuid4().hex[:12].upper()}"
        
        # Update positions
        self._update_position(order, exec_price)
        
        # Log trade
        trade = Trade(
            id=None,
            timestamp=datetime.now(),
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=exec_price,
            order_id=order_id,
            pnl=None,  # P&L calculated on position close
            is_paper=True,
        )
        self._data_store.log_trade(trade)
        
        return OrderResult(
            order_id=order_id,
            status="COMPLETE",
            filled_qty=order.quantity,
            filled_price=exec_price,
            message="Paper order executed successfully",
        )

    def _update_position(self, order: Order, exec_price: float) -> None:
        """Update position after order execution.
        
        Args:
            order: Executed order.
            exec_price: Execution price.
        """
        # Get existing position
        positions = self._data_store.get_positions()
        existing = next((p for p in positions if p.symbol == order.symbol), None)
        
        if order.side == "BUY":
            if existing:
                # Add to existing position
                total_qty = existing.quantity + order.quantity
                total_value = (existing.average_price * existing.quantity) + (exec_price * order.quantity)
                new_avg = total_value / total_qty if total_qty > 0 else 0
                
                new_position = Position(
                    symbol=order.symbol,
                    quantity=total_qty,
                    average_price=new_avg,
                    ltp=exec_price,
                    pnl=(exec_price - new_avg) * total_qty,
                    pnl_percent=((exec_price - new_avg) / new_avg * 100) if new_avg > 0 else 0,
                    product=order.product,
                )
            else:
                # New position
                new_position = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    average_price=exec_price,
                    ltp=exec_price,
                    pnl=0.0,
                    pnl_percent=0.0,
                    product=order.product,
                )
            self._data_store.save_position(new_position)
        else:
            # SELL
            if existing:
                remaining_qty = existing.quantity - order.quantity
                if remaining_qty <= 0:
                    # Position closed
                    self._data_store.delete_position(order.symbol)
                else:
                    # Reduce position
                    new_position = Position(
                        symbol=order.symbol,
                        quantity=remaining_qty,
                        average_price=existing.average_price,
                        ltp=exec_price,
                        pnl=(exec_price - existing.average_price) * remaining_qty,
                        pnl_percent=((exec_price - existing.average_price) / existing.average_price * 100) if existing.average_price > 0 else 0,
                        product=order.product,
                    )
                    self._data_store.save_position(new_position)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order (always fails for paper - orders execute immediately).
        
        Args:
            order_id: Order ID.
            
        Returns:
            False always (paper orders execute immediately).
        """
        return False

    def get_positions(self) -> list[Position]:
        """Get all open positions.
        
        Returns:
            List of positions from database.
        """
        positions = self._data_store.get_positions()
        
        # Update LTP and P&L for each position
        updated_positions = []
        for pos in positions:
            current_price = self._get_simulated_price(pos.symbol)
            pnl = (current_price - pos.average_price) * pos.quantity
            pnl_percent = ((current_price - pos.average_price) / pos.average_price * 100) if pos.average_price > 0 else 0
            
            updated_positions.append(Position(
                symbol=pos.symbol,
                quantity=pos.quantity,
                average_price=pos.average_price,
                ltp=current_price,
                pnl=pnl,
                pnl_percent=pnl_percent,
                product=pos.product,
            ))
        
        return updated_positions

    def get_balance(self) -> Balance:
        """Get account balance.
        
        Returns:
            Current balance information.
        """
        # Calculate used margin from positions
        positions = self._data_store.get_positions()
        used_margin = sum(
            pos.average_price * pos.quantity
            for pos in positions
        )
        
        return Balance(
            available_cash=self._balance,
            used_margin=used_margin,
            total_value=self._balance + used_margin,
        )

    def reset(self) -> None:
        """Reset paper trading to initial state.
        
        Clears all positions and resets balance to starting amount.
        """
        # Clear all positions
        positions = self._data_store.get_positions()
        for pos in positions:
            self._data_store.delete_position(pos.symbol)
        
        # Reset balance
        self._balance = self._starting_balance
        self._save_balance()

    def get_current_balance(self) -> float:
        """Get the current cash balance.
        
        Returns:
            Current available cash balance.
        """
        return self._balance
