"""Trade data model."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Trade(BaseModel):
    """Represents an executed trade."""

    id: Optional[int] = Field(default=None, description="Database ID")
    timestamp: datetime = Field(..., description="Trade execution timestamp")
    symbol: str = Field(..., min_length=1, description="Trading symbol")
    side: str = Field(..., description="Trade side (BUY/SELL)")
    quantity: int = Field(..., gt=0, description="Trade quantity")
    price: float = Field(..., ge=0, description="Execution price")
    order_id: str = Field(..., description="Associated order ID")
    pnl: Optional[float] = Field(default=None, description="Realized P&L")
    is_paper: bool = Field(default=False, description="Paper trade flag")

    model_config = {"frozen": True}
