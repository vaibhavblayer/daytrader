"""Technical Verification Agent for indicator analysis.

This agent verifies technical indicators and provides
AI-powered trading recommendations.
"""

import os
from typing import Optional

from agents import Agent

from daytrader.agents.base import create_agent, run_agent_sync

# Disable tracing to avoid noisy 503 errors
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

# Default model with medium reasoning
DEFAULT_TECHNICAL_MODEL = "gpt-5.2"


TECHNICAL_ANALYST_INSTRUCTIONS = """You are an expert Indian stock market technical analyst with deep knowledge of:
- Price action and candlestick patterns
- Technical indicators (RSI, MACD, Bollinger Bands, SuperTrend, ADX, Stochastic, Fibonacci)
- Support/Resistance levels and trend analysis
- Risk management and position sizing

Your role is to verify technical indicator readings and provide actionable trading advice.

When analyzing indicators:
1. VERIFY if the signals are aligned or conflicting
2. Assess the STRENGTH of the setup (strong/moderate/weak)
3. Identify the overall BIAS (Bullish/Bearish/Neutral) with confidence percentage
4. Provide specific TRADE SETUP if conditions are favorable:
   - Entry price or zone
   - Stop-loss level with reasoning
   - Target levels (T1, T2)
   - Risk:Reward ratio
5. Highlight KEY RISKS to watch

Be concise and actionable. Focus on what matters for the trader.
Use Indian market context (NSE/BSE, market hours 9:15 AM - 3:30 PM IST, MIS square-off at 3:15 PM).
"""


class TechnicalVerificationAgent:
    """Agent for verifying technical indicators and providing trade recommendations.
    
    Uses GPT with medium reasoning to analyze indicator data and provide
    AI-powered verification and trading advice.
    """
    
    def __init__(self, model: Optional[str] = None):
        """Initialize the Technical Verification Agent.
        
        Args:
            model: Optional model override. Defaults to o4-mini.
        """
        self.model = model or DEFAULT_TECHNICAL_MODEL
        self._agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create the underlying agent."""
        return create_agent(
            name="Technical Verification Agent",
            instructions=TECHNICAL_ANALYST_INSTRUCTIONS,
            tools=[],  # No tools needed, just analysis
            model=self.model,
        )
    
    def verify(
        self,
        symbol: str,
        timeframe: str,
        price: float,
        candle_count: int,
        indicators: dict,
    ) -> str:
        """Verify technical indicators and provide trading advice.
        
        Args:
            symbol: Stock symbol.
            timeframe: Candle timeframe (1min, 5min, 15min, 1hour, 1day).
            price: Current price.
            candle_count: Number of candles analyzed.
            indicators: Dictionary of indicator results.
            
        Returns:
            AI analysis and recommendations.
        """
        # Build indicator summary
        ind_lines = []
        
        if "rsi" in indicators:
            rsi = indicators["rsi"]
            ind_lines.append(f"RSI(14): {rsi['value']:.1f} - {rsi['signal']}")
        
        if "macd" in indicators:
            macd = indicators["macd"]
            ind_lines.append(
                f"MACD: {macd['macd']:.4f}, Signal: {macd['signal']:.4f}, "
                f"Histogram: {macd['histogram']:.4f} - {macd['interpretation']}"
            )
        
        if "bb" in indicators:
            bb = indicators["bb"]
            ind_lines.append(
                f"Bollinger Bands: Upper={bb['upper']:.2f}, Middle={bb['middle']:.2f}, "
                f"Lower={bb['lower']:.2f} - {bb['signal']}"
            )
        
        if "supertrend" in indicators:
            st = indicators["supertrend"]
            ind_lines.append(f"SuperTrend: {st['value']:.2f} - {st['signal']}")
        
        if "adx" in indicators:
            adx = indicators["adx"]
            ind_lines.append(
                f"ADX: {adx['adx']:.1f}, +DI: {adx['plus_di']:.1f}, -DI: {adx['minus_di']:.1f} - "
                f"{adx['trend_strength']} ({adx['direction']})"
            )
        
        if "stoch" in indicators:
            stoch = indicators["stoch"]
            ind_lines.append(
                f"Stochastic: %K={stoch['k']:.1f}, %D={stoch['d']:.1f} - {stoch['signal']}"
            )
        
        if "fib" in indicators:
            fib = indicators["fib"]
            support = fib.get("nearest_support")
            resistance = fib.get("nearest_resistance")
            support_str = f"{support[1]:.2f}" if support else "N/A"
            resist_str = f"{resistance[1]:.2f}" if resistance else "N/A"
            ind_lines.append(
                f"Fibonacci: Trend={fib['trend']}, Support={support_str}, Resistance={resist_str}"
            )
        
        if "pivot" in indicators:
            pivot = indicators["pivot"]
            levels = pivot["levels"]
            ind_lines.append(
                f"Pivot: P={levels['Pivot']:.2f}, R1={levels['R1']:.2f}, S1={levels['S1']:.2f} - "
                f"{pivot['position']}"
            )
        
        if "patterns" in indicators:
            patterns = indicators["patterns"]
            if patterns["detected"]:
                pattern_str = ", ".join(
                    [f"{p['pattern']}({p['signal']})" for p in patterns["detected"]]
                )
                ind_lines.append(f"Candlestick Patterns: {pattern_str}")
        
        if "ema" in indicators:
            ema = indicators["ema"]
            ind_lines.append(
                f"EMA: 9-day={ema['ema_9']:.2f}, 21-day={ema['ema_21']:.2f} - {ema['trend']} Trend"
            )
        
        if "atr" in indicators:
            atr = indicators["atr"]
            ind_lines.append(
                f"ATR(14): {atr['value']:.2f} ({atr['percent']:.1f}%) - {atr['volatility']} Volatility"
            )
        
        indicator_text = "\n".join(ind_lines) if ind_lines else "No indicators calculated"
        
        prompt = f"""Analyze the following technical indicators for {symbol} on {timeframe} timeframe:

Current Price: ₹{price:.2f}
Candles Analyzed: {candle_count}

Technical Indicators:
{indicator_text}

Provide:
1. VERIFICATION: Are signals aligned or conflicting?
2. BIAS: Overall bias (Bullish/Bearish/Neutral) with confidence %
3. TRADE SETUP: Entry, Stop-Loss, Target (if favorable setup exists)
4. RISK: Key risks to watch
5. RECOMMENDATION: Clear action (BUY/SELL/WAIT) with brief reasoning

Keep response under 200 words. Be specific with price levels."""

        return run_agent_sync(self._agent, prompt)


def _cleanup_markdown(text: str) -> str:
    """Convert markdown formatting to Rich-compatible plain text.
    
    Args:
        text: Text with markdown formatting.
        
    Returns:
        Clean text with Rich formatting.
    """
    import re
    
    # Convert **bold** to [bold]...[/bold]
    text = re.sub(r'\*\*([^*]+)\*\*', r'[bold]\1[/bold]', text)
    
    # Convert *italic* to [italic]...[/italic]
    text = re.sub(r'\*([^*]+)\*', r'[italic]\1[/italic]', text)
    
    # Convert `code` to [cyan]...[/cyan]
    text = re.sub(r'`([^`]+)`', r'[cyan]\1[/cyan]', text)
    
    # Remove markdown headers (# ## ###)
    text = re.sub(r'^#{1,3}\s+', '', text, flags=re.MULTILINE)
    
    # Convert markdown bullet points to simple dashes
    text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)
    
    # Highlight price levels (₹ followed by numbers)
    text = re.sub(r'(₹[\d,]+\.?\d*)', r'[cyan]\1[/cyan]', text)
    
    # Highlight BUY/SELL/WAIT recommendations
    text = re.sub(r'\b(BUY)\b', r'[green]\1[/green]', text)
    text = re.sub(r'\b(SELL)\b', r'[red]\1[/red]', text)
    text = re.sub(r'\b(WAIT)\b', r'[yellow]\1[/yellow]', text)
    
    # Highlight Bullish/Bearish
    text = re.sub(r'\b(Bullish|bullish|BULLISH)\b', r'[green]\1[/green]', text)
    text = re.sub(r'\b(Bearish|bearish|BEARISH)\b', r'[red]\1[/red]', text)
    
    return text


def get_ai_verification(
    symbol: str,
    timeframe: str,
    results: dict,
    model: Optional[str] = None,
) -> str:
    """Get AI verification for technical analysis results.
    
    Args:
        symbol: Stock symbol.
        timeframe: Candle timeframe.
        results: Results from calculate_indicators_for_symbol().
        model: Optional model override.
        
    Returns:
        AI analysis text with Rich formatting.
    """
    agent = TechnicalVerificationAgent(model=model)
    
    response = agent.verify(
        symbol=symbol,
        timeframe=timeframe,
        price=results.get("last_price", 0),
        candle_count=results.get("candle_count", 0),
        indicators=results.get("indicators", {}),
    )
    
    return _cleanup_markdown(response)
