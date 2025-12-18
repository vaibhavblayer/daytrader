"""Candle (OHLCV) data model."""

from datetime import datetime
from pydantic import BaseModel, Field


class Candle(BaseModel):
    """Represents a single OHLCV candle."""

    timestamp: datetime = Field(..., description="Candle timestamp")
    open: float = Field(..., ge=0, description="Opening price")
    high: float = Field(..., ge=0, description="High price")
    low: float = Field(..., ge=0, description="Low price")
    close: float = Field(..., ge=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")

    model_config = {"frozen": True}
