"""JournalEntry data model."""

from datetime import date as date_type
from typing import Optional
from pydantic import BaseModel, Field


class JournalEntry(BaseModel):
    """Represents a daily trading journal entry."""

    date: date_type = Field(..., description="Journal entry date")
    trades_count: int = Field(..., ge=0, description="Number of trades")
    total_pnl: float = Field(..., description="Total P&L for the day")
    win_rate: float = Field(..., ge=0, le=100, description="Win rate percentage")
    notes: Optional[str] = Field(default=None, description="User notes")
    ai_insights: Optional[str] = Field(default=None, description="AI-generated insights")

    model_config = {"frozen": True}
