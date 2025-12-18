"""Data Analyst Agent for technical analysis.

This agent analyzes price data, calculates technical indicators,
and identifies chart patterns to provide trading insights.
"""

from pathlib import Path
from typing import Optional

from agents import Agent, function_tool

from daytrader.agents.base import create_agent, run_agent_sync
from daytrader.db.store import DataStore
from daytrader.tools.database import query_candles
from daytrader.tools.indicators import calculate_indicators, find_support_resistance


# Default database path
DEFAULT_DB_PATH = Path.home() / ".config" / "daytrader" / "daytrader.db"


DATA_ANALYST_INSTRUCTIONS = """You are a technical analyst specializing in Indian equity markets.
Your role is to analyze price data and provide actionable trading insights.

When analyzing a stock:
1. Query historical price data from the database
2. Calculate relevant technical indicators (RSI, MACD, Bollinger Bands, etc.)
3. Identify support and resistance levels
4. Look for chart patterns and trends

Always provide:
- Current technical setup with specific indicator values
- Key support and resistance levels
- Trading signals (bullish/bearish/neutral)
- Suggested entry/exit levels with rationale

Be specific with numbers and levels. Explain your reasoning clearly.
Focus on actionable insights for day trading.
"""


@function_tool
def query_candles_tool(
    symbol: str,
    timeframe: str = "1day",
    days: int = 30,
) -> dict:
    """Query historical price data from the database.
    
    Use this tool to get OHLCV (Open, High, Low, Close, Volume) data
    for technical analysis.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE", "TCS").
        timeframe: Candle timeframe - "1min", "5min", "15min", "1hour", "1day".
        days: Number of days of data to retrieve.
        
    Returns:
        Price data with candles and latest values.
    """
    return query_candles(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
    )


@function_tool
def calculate_indicators_tool(
    symbol: str,
    indicators: Optional[list[str]] = None,
    timeframe: str = "1day",
    days: int = 100,
) -> dict:
    """Calculate technical indicators for a symbol.
    
    Use this tool to calculate RSI, MACD, EMA, SMA, Bollinger Bands,
    ATR, and VWAP indicators.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE", "TCS").
        indicators: List of indicators to calculate. Options:
                   "rsi", "macd", "ema", "sma", "bb", "atr", "vwap".
                   If None, calculates all.
        timeframe: Candle timeframe (default "1day").
        days: Number of days of data to use.
        
    Returns:
        Calculated indicator values and trading signals.
    """
    return calculate_indicators(
        symbol=symbol,
        indicators=indicators,
        timeframe=timeframe,
        days=days,
    )


@function_tool
def find_support_resistance_tool(
    symbol: str,
    timeframe: str = "1day",
    days: int = 60,
    num_levels: int = 3,
) -> dict:
    """Find support and resistance levels for a symbol.
    
    Use this tool to identify key price levels based on historical
    data and pivot points.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE", "TCS").
        timeframe: Candle timeframe (default "1day").
        days: Number of days of data to analyze.
        num_levels: Number of support/resistance levels to return.
        
    Returns:
        Support levels, resistance levels, and pivot points.
    """
    return find_support_resistance(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        num_levels=num_levels,
    )


@function_tool
def find_patterns_tool(
    symbol: str,
    timeframe: str = "1day",
    days: int = 60,
) -> dict:
    """Identify chart patterns in price data.
    
    Use this tool to look for common chart patterns like
    double tops/bottoms, head and shoulders, triangles, etc.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE", "TCS").
        timeframe: Candle timeframe (default "1day").
        days: Number of days of data to analyze.
        
    Returns:
        Identified patterns with descriptions.
    """
    # Get candle data
    candle_data = query_candles(symbol=symbol, timeframe=timeframe, days=days)
    
    if candle_data.get("error") or not candle_data.get("candles"):
        return {
            "symbol": symbol.upper(),
            "patterns": [],
            "error": candle_data.get("error", "No data available"),
        }
    
    candles = candle_data["candles"]
    patterns = []
    
    if len(candles) < 10:
        return {
            "symbol": symbol.upper(),
            "patterns": [],
            "trend": "insufficient_data",
            "error": None,
        }
    
    # Extract price data
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    
    # Determine overall trend
    recent_closes = closes[-20:] if len(closes) >= 20 else closes
    if len(recent_closes) >= 2:
        if recent_closes[-1] > recent_closes[0] * 1.05:
            trend = "uptrend"
        elif recent_closes[-1] < recent_closes[0] * 0.95:
            trend = "downtrend"
        else:
            trend = "sideways"
    else:
        trend = "unknown"
    
    # Look for higher highs and higher lows (uptrend confirmation)
    if len(highs) >= 10:
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        # Check for higher highs
        higher_highs = all(
            recent_highs[i] >= recent_highs[i-1] * 0.99
            for i in range(1, len(recent_highs))
        )
        
        # Check for higher lows
        higher_lows = all(
            recent_lows[i] >= recent_lows[i-1] * 0.99
            for i in range(1, len(recent_lows))
        )
        
        if higher_highs and higher_lows:
            patterns.append({
                "name": "Higher Highs and Higher Lows",
                "type": "bullish",
                "description": "Price making higher highs and higher lows - bullish trend",
            })
    
    # Look for potential double bottom
    if len(lows) >= 20:
        min_idx = lows[-20:].index(min(lows[-20:]))
        if 5 <= min_idx <= 15:  # Not at edges
            left_min = min(lows[-20:-20+min_idx])
            right_min = min(lows[-20+min_idx:])
            if abs(left_min - right_min) / left_min < 0.02:  # Within 2%
                patterns.append({
                    "name": "Potential Double Bottom",
                    "type": "bullish",
                    "description": "Two similar lows detected - potential reversal pattern",
                    "level": round(min(left_min, right_min), 2),
                })
    
    # Look for potential double top
    if len(highs) >= 20:
        max_idx = highs[-20:].index(max(highs[-20:]))
        if 5 <= max_idx <= 15:  # Not at edges
            left_max = max(highs[-20:-20+max_idx])
            right_max = max(highs[-20+max_idx:])
            if abs(left_max - right_max) / left_max < 0.02:  # Within 2%
                patterns.append({
                    "name": "Potential Double Top",
                    "type": "bearish",
                    "description": "Two similar highs detected - potential reversal pattern",
                    "level": round(max(left_max, right_max), 2),
                })
    
    # Check for consolidation/range
    if len(closes) >= 10:
        recent_range = max(highs[-10:]) - min(lows[-10:])
        avg_price = sum(closes[-10:]) / 10
        if recent_range / avg_price < 0.05:  # Less than 5% range
            patterns.append({
                "name": "Consolidation",
                "type": "neutral",
                "description": "Price consolidating in tight range - breakout expected",
                "range_high": round(max(highs[-10:]), 2),
                "range_low": round(min(lows[-10:]), 2),
            })
    
    return {
        "symbol": symbol.upper(),
        "patterns": patterns,
        "trend": trend,
        "data_points": len(candles),
        "error": None,
    }


class DataAnalystAgent:
    """Agent for technical analysis of price data.
    
    This agent queries price data from the database, calculates
    technical indicators, and identifies patterns to provide
    trading insights.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the Data Analyst Agent.
        
        Args:
            db_path: Optional path to database.
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self._store = DataStore(self.db_path)
        self._agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the underlying agent."""
        return create_agent(
            name="Data Analyst Agent",
            instructions=DATA_ANALYST_INSTRUCTIONS,
            tools=[
                query_candles_tool,
                calculate_indicators_tool,
                find_support_resistance_tool,
                find_patterns_tool,
            ],
        )
    
    def analyze(self, symbol: str, query: Optional[str] = None) -> str:
        """Analyze a stock and return technical insights.
        
        Args:
            symbol: Stock symbol to analyze.
            query: Optional specific question about the stock.
            
        Returns:
            Technical analysis as a string.
        """
        if query:
            prompt = f"""Analyze {symbol.upper()} and answer this question: {query}

Use the available tools to:
1. Get price data
2. Calculate relevant indicators
3. Find support/resistance levels
4. Identify any patterns

Provide specific numbers and actionable insights."""
        else:
            prompt = f"""Provide a comprehensive technical analysis of {symbol.upper()}.

Use the available tools to:
1. Get recent price data
2. Calculate all relevant indicators
3. Find support and resistance levels
4. Identify any chart patterns

Include:
- Current technical setup
- Key levels to watch
- Trading signals
- Suggested entry/exit points"""
        
        return run_agent_sync(self._agent, prompt)
    
    def ask(self, query: str) -> str:
        """Ask the analyst agent a question.
        
        Args:
            query: Question to ask.
            
        Returns:
            Agent's response.
        """
        return run_agent_sync(self._agent, query)
