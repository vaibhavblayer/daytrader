"""Alert data model."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class Alert(BaseModel):
    """Represents a price/indicator alert."""

    id: Optional[int] = Field(default=None, description="Database ID")
    symbol: str = Field(..., min_length=1, description="Trading symbol")
    condition: str = Field(
        ..., min_length=1, description="Alert condition (e.g., 'price > 1500')"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Alert creation timestamp"
    )
    triggered: bool = Field(default=False, description="Whether alert has triggered")

    model_config = {"frozen": True}
