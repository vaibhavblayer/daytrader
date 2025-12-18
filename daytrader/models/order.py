"""Order and OrderResult data models."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class Order(BaseModel):
    """Represents an order to be placed."""

    symbol: str = Field(..., min_length=1, description="Trading symbol")
    side: Literal["BUY", "SELL"] = Field(..., description="Order side")
    quantity: int = Field(..., gt=0, description="Order quantity")
    order_type: Literal["MARKET", "LIMIT", "SL", "SL-M"] = Field(
        ..., description="Order type"
    )
    product: Literal["MIS", "CNC"] = Field(
        default="MIS", description="Product type (MIS=intraday, CNC=delivery)"
    )
    variety: Literal["NORMAL", "AMO"] = Field(
        default="NORMAL", description="Order variety (NORMAL=regular, AMO=after market)"
    )
    price: Optional[float] = Field(
        default=None, ge=0, description="Limit price (for LIMIT orders)"
    )
    trigger_price: Optional[float] = Field(
        default=None, ge=0, description="Trigger price (for SL orders)"
    )

    model_config = {"frozen": True}


class OrderResult(BaseModel):
    """Represents the result of an order placement."""

    order_id: str = Field(..., description="Unique order identifier")
    status: str = Field(..., description="Order status")
    filled_qty: int = Field(..., ge=0, description="Filled quantity")
    filled_price: float = Field(..., ge=0, description="Average filled price")
    message: str = Field(default="", description="Status message")

    model_config = {"frozen": True}
