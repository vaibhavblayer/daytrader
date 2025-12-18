"""Base broker interface for DayTrader."""

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

from daytrader.models import Candle, Order, OrderResult, Position


class Quote(BaseModel):
    """Represents a real-time quote for a symbol."""

    symbol: str = Field(..., description="Trading symbol")
    ltp: float = Field(..., ge=0, description="Last traded price")
    change: float = Field(..., description="Price change from previous close")
    change_percent: float = Field(..., description="Percentage change")
    volume: int = Field(..., ge=0, description="Trading volume")
    open: float = Field(..., ge=0, description="Opening price")
    high: float = Field(..., ge=0, description="Day high")
    low: float = Field(..., ge=0, description="Day low")
    close: float = Field(..., ge=0, description="Previous close")

    model_config = {"frozen": True}


class Balance(BaseModel):
    """Represents account balance information."""

    available_cash: float = Field(..., description="Available cash for trading")
    used_margin: float = Field(..., ge=0, description="Margin currently in use")
    total_value: float = Field(..., description="Total portfolio value")

    model_config = {"frozen": True}


class BaseBroker(ABC):
    """Abstract base class for broker implementations.
    
    All broker implementations (Angel One, Paper Trading, etc.) must
    inherit from this class and implement all abstract methods.
    """

    @abstractmethod
    def login(self) -> bool:
        """Authenticate with the broker.
        
        Returns:
            True if authentication successful, False otherwise.
        """
        pass

    @abstractmethod
    def logout(self) -> bool:
        """Logout and invalidate session.
        
        Returns:
            True if logout successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """Get real-time quote for a symbol.
        
        Args:
            symbol: Trading symbol.
            
        Returns:
            Quote with current market data.
            
        Raises:
            ValueError: If symbol is invalid.
        """
        pass

    @abstractmethod
    def get_historical(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        interval: str,
    ) -> list[Candle]:
        """Get historical OHLCV data.
        
        Args:
            symbol: Trading symbol.
            from_date: Start date.
            to_date: End date.
            interval: Candle interval (1min, 5min, 15min, 1hour, 1day).
            
        Returns:
            List of candles for the date range.
            
        Raises:
            ValueError: If symbol or interval is invalid.
        """
        pass

    @abstractmethod
    def place_order(self, order: Order) -> OrderResult:
        """Place an order.
        
        Args:
            order: Order to place.
            
        Returns:
            OrderResult with execution details.
            
        Raises:
            ValueError: If order parameters are invalid.
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.
        
        Args:
            order_id: ID of the order to cancel.
            
        Returns:
            True if cancellation successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all open positions.
        
        Returns:
            List of current positions.
        """
        pass

    @abstractmethod
    def get_balance(self) -> Balance:
        """Get account balance information.
        
        Returns:
            Balance with available funds and margin info.
        """
        pass

    @abstractmethod
    def is_authenticated(self) -> bool:
        """Check if currently authenticated.
        
        Returns:
            True if authenticated, False otherwise.
        """
        pass
