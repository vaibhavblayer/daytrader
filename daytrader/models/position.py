"""Position data model."""

from pydantic import BaseModel, Field


class Position(BaseModel):
    """Represents an open trading position."""

    symbol: str = Field(..., min_length=1, description="Trading symbol")
    quantity: int = Field(..., description="Position quantity (negative for short)")
    average_price: float = Field(..., ge=0, description="Average entry price")
    ltp: float = Field(..., ge=0, description="Last traded price")
    pnl: float = Field(..., description="Profit/Loss amount")
    pnl_percent: float = Field(..., description="Profit/Loss percentage")
    product: str = Field(..., description="Product type (MIS/CNC)")

    model_config = {"frozen": True}
