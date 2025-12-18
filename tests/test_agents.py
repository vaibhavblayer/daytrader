"""Property-based tests for AI agents.

**Feature: day-trading-cli**
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from daytrader.agents.news import (
    VALID_SENTIMENTS,
    analyze_sentiment,
    classify_sentiment,
)


class TestSentimentClassification:
    """
    **Feature: day-trading-cli, Property 25: News Sentiment Classification**
    **Validates: Requirements 11.2**
    
    *For any* news analysis, the sentiment should be classified as one of:
    bullish, bearish, neutral.
    """

    @given(
        text=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip() != ""),
    )
    @settings(max_examples=100)
    def test_sentiment_always_valid_category(self, text: str):
        """
        *For any* text input, sentiment classification should return
        one of the valid categories: bullish, bearish, or neutral.
        """
        result = analyze_sentiment(text)
        
        assert "sentiment" in result, "Result missing 'sentiment' key"
        assert result["sentiment"] in VALID_SENTIMENTS, (
            f"Invalid sentiment '{result['sentiment']}', "
            f"expected one of {VALID_SENTIMENTS}"
        )

    @given(
        text=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip() != ""),
    )
    @settings(max_examples=100)
    def test_sentiment_confidence_in_range(self, text: str):
        """
        *For any* text input, sentiment confidence should be between 0 and 1.
        """
        result = analyze_sentiment(text)
        
        assert "confidence" in result, "Result missing 'confidence' key"
        assert 0 <= result["confidence"] <= 1, (
            f"Confidence {result['confidence']} out of range [0, 1]"
        )

    @given(
        text=st.text(min_size=1, max_size=1000).filter(lambda x: x.strip() != ""),
    )
    @settings(max_examples=50)
    def test_classify_sentiment_returns_tuple(self, text: str):
        """
        *For any* text input, classify_sentiment should return a tuple
        of (sentiment, confidence).
        """
        sentiment, confidence = classify_sentiment(text)
        
        assert sentiment in VALID_SENTIMENTS, (
            f"Invalid sentiment '{sentiment}'"
        )
        assert 0 <= confidence <= 1, (
            f"Confidence {confidence} out of range [0, 1]"
        )

    def test_bullish_keywords_detected(self):
        """Bullish keywords should result in bullish sentiment."""
        bullish_texts = [
            "Stock surges on strong earnings beat",
            "Analysts upgrade rating to buy with positive outlook",
            "Company reports record profit growth",
            "Stock rallies on acquisition news",
        ]
        
        for text in bullish_texts:
            result = analyze_sentiment(text)
            assert result["sentiment"] == "bullish", (
                f"Expected bullish for '{text}', got {result['sentiment']}"
            )
            assert result["bullish_signals"] > 0

    def test_bearish_keywords_detected(self):
        """Bearish keywords should result in bearish sentiment."""
        bearish_texts = [
            "Stock plunges on earnings miss",
            "Analysts downgrade rating to sell with negative outlook",
            "Company reports significant loss",
            "Stock crashes on debt concerns",
        ]
        
        for text in bearish_texts:
            result = analyze_sentiment(text)
            assert result["sentiment"] == "bearish", (
                f"Expected bearish for '{text}', got {result['sentiment']}"
            )
            assert result["bearish_signals"] > 0

    def test_neutral_for_no_keywords(self):
        """Text without sentiment keywords should be neutral."""
        neutral_texts = [
            "The company held its annual meeting today",
            "Trading volume was average",
            "The stock closed unchanged",
        ]
        
        for text in neutral_texts:
            result = analyze_sentiment(text)
            assert result["sentiment"] == "neutral", (
                f"Expected neutral for '{text}', got {result['sentiment']}"
            )

    @given(
        bullish_count=st.integers(min_value=1, max_value=5),
        bearish_count=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=50)
    def test_more_bullish_signals_means_bullish(self, bullish_count: int, bearish_count: int):
        """
        *For any* text with more bullish than bearish signals,
        sentiment should be bullish.
        """
        if bullish_count <= bearish_count:
            return  # Skip when not more bullish
        
        # Construct text with specific keyword counts
        bullish_words = ["surge", "rally", "gain", "rise", "jump"]
        bearish_words = ["fall", "drop", "decline", "plunge", "crash"]
        
        text_parts = []
        for i in range(bullish_count):
            text_parts.append(f"Stock {bullish_words[i % len(bullish_words)]}")
        for i in range(bearish_count):
            text_parts.append(f"Some {bearish_words[i % len(bearish_words)]}")
        
        text = ". ".join(text_parts)
        result = analyze_sentiment(text)
        
        assert result["sentiment"] == "bullish", (
            f"Expected bullish with {bullish_count} bullish vs {bearish_count} bearish signals"
        )

    @given(
        bearish_count=st.integers(min_value=1, max_value=5),
        bullish_count=st.integers(min_value=0, max_value=5),
    )
    @settings(max_examples=50)
    def test_more_bearish_signals_means_bearish(self, bearish_count: int, bullish_count: int):
        """
        *For any* text with more bearish than bullish signals,
        sentiment should be bearish.
        """
        if bearish_count <= bullish_count:
            return  # Skip when not more bearish
        
        # Construct text with specific keyword counts
        bullish_words = ["surge", "rally", "gain", "rise", "jump"]
        bearish_words = ["fall", "drop", "decline", "plunge", "crash"]
        
        text_parts = []
        for i in range(bearish_count):
            text_parts.append(f"Stock {bearish_words[i % len(bearish_words)]}")
        for i in range(bullish_count):
            text_parts.append(f"Some {bullish_words[i % len(bullish_words)]}")
        
        text = ". ".join(text_parts)
        result = analyze_sentiment(text)
        
        assert result["sentiment"] == "bearish", (
            f"Expected bearish with {bearish_count} bearish vs {bullish_count} bullish signals"
        )



from daytrader.agents.orchestrator import (
    classify_query,
    is_trading_query,
    extract_symbol,
    RESEARCH_KEYWORDS,
    ANALYST_KEYWORDS,
    NEWS_KEYWORDS,
    TRADING_KEYWORDS,
)


class TestOrchestratorRouting:
    """
    **Feature: day-trading-cli, Property 27: Orchestrator Routing Consistency**
    **Validates: Requirements 12.2**
    
    *For any* research-related query, the Orchestrator should route to 
    the Research Agent.
    """

    @given(
        keyword=st.sampled_from(RESEARCH_KEYWORDS),
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=3,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
    )
    @settings(max_examples=50)
    def test_research_queries_route_to_research_agent(self, keyword: str, symbol: str):
        """
        *For any* query containing research keywords, the orchestrator
        should include 'research' in the routing.
        """
        query = f"Tell me about {symbol} {keyword}"
        agents = classify_query(query)
        
        assert "research" in agents, (
            f"Query '{query}' with keyword '{keyword}' should route to research agent"
        )

    @given(
        keyword=st.sampled_from(ANALYST_KEYWORDS),
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=3,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
    )
    @settings(max_examples=50)
    def test_analyst_queries_route_to_analyst_agent(self, keyword: str, symbol: str):
        """
        *For any* query containing analyst keywords, the orchestrator
        should include 'analyst' in the routing.
        """
        query = f"What is the {keyword} for {symbol}?"
        agents = classify_query(query)
        
        assert "analyst" in agents, (
            f"Query '{query}' with keyword '{keyword}' should route to analyst agent"
        )

    @given(
        keyword=st.sampled_from(NEWS_KEYWORDS),
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=3,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
    )
    @settings(max_examples=50)
    def test_news_queries_route_to_news_agent(self, keyword: str, symbol: str):
        """
        *For any* query containing news keywords, the orchestrator
        should include 'news' in the routing.
        """
        query = f"What is the {keyword} for {symbol}?"
        agents = classify_query(query)
        
        assert "news" in agents, (
            f"Query '{query}' with keyword '{keyword}' should route to news agent"
        )

    @given(
        keyword=st.sampled_from(TRADING_KEYWORDS),
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=3,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
    )
    @settings(max_examples=50)
    def test_trading_queries_route_to_trading_agent(self, keyword: str, symbol: str):
        """
        *For any* query containing trading keywords, the orchestrator
        should include 'trading' in the routing.
        """
        query = f"{keyword} {symbol}"
        agents = classify_query(query)
        
        assert "trading" in agents, (
            f"Query '{query}' with keyword '{keyword}' should route to trading agent"
        )

    @given(
        query=st.text(min_size=1, max_size=200).filter(lambda x: x.strip() != ""),
    )
    @settings(max_examples=100)
    def test_classify_always_returns_valid_agents(self, query: str):
        """
        *For any* query, classify_query should return a non-empty list
        of valid agent types.
        """
        agents = classify_query(query)
        
        assert len(agents) > 0, "classify_query should always return at least one agent"
        
        valid_agents = {"research", "analyst", "news", "trading", "orchestrator"}
        for agent in agents:
            assert agent in valid_agents, f"Invalid agent type: {agent}"

    def test_research_specific_queries(self):
        """Research-specific queries should route to research agent."""
        research_queries = [
            "Research RELIANCE company",
            "What do analysts think about TCS?",
            "Tell me about INFY fundamentals",
            "Company profile for HDFCBANK",
        ]
        
        for query in research_queries:
            agents = classify_query(query)
            assert "research" in agents, f"Query '{query}' should route to research"

    def test_analyst_specific_queries(self):
        """Technical analysis queries should route to analyst agent."""
        analyst_queries = [
            "What is the RSI for RELIANCE?",
            "Show me MACD for TCS",
            "Find support and resistance for INFY",
            "Is HDFCBANK oversold?",
            "Technical analysis of SBIN",
        ]
        
        for query in analyst_queries:
            agents = classify_query(query)
            assert "analyst" in agents, f"Query '{query}' should route to analyst"

    def test_news_specific_queries(self):
        """News queries should route to news agent."""
        news_queries = [
            "What's the latest news on RELIANCE?",
            "Any recent announcements for TCS?",
            "Market sentiment for INFY",
            "What's happening with HDFCBANK today?",
        ]
        
        for query in news_queries:
            agents = classify_query(query)
            assert "news" in agents, f"Query '{query}' should route to news"

    def test_trading_specific_queries(self):
        """Trading queries should route to trading agent."""
        trading_queries = [
            "Buy 100 shares of RELIANCE",
            "Sell TCS at market price",
            "Show my positions",
            "What's my balance?",
            "Exit all positions",
        ]
        
        for query in trading_queries:
            agents = classify_query(query)
            assert "trading" in agents, f"Query '{query}' should route to trading"

    def test_is_trading_query_detection(self):
        """is_trading_query should correctly identify trading actions."""
        trading_queries = [
            "Buy RELIANCE",
            "Sell 100 TCS",
            "Execute trade for INFY",
            "Place order for HDFCBANK",
            "Exit position in SBIN",
        ]
        
        for query in trading_queries:
            assert is_trading_query(query), f"'{query}' should be detected as trading query"
        
        non_trading_queries = [
            "What is RSI for RELIANCE?",
            "Research TCS company",
            "News about INFY",
        ]
        
        for query in non_trading_queries:
            assert not is_trading_query(query), f"'{query}' should not be detected as trading query"

    def test_symbol_extraction(self):
        """extract_symbol should correctly extract stock symbols."""
        test_cases = [
            ("Buy RELIANCE", "RELIANCE"),
            ("What is RSI for TCS?", "TCS"),
            ("Research INFY company", "INFY"),
            ("HDFCBANK technical analysis", "HDFCBANK"),
        ]
        
        for query, expected in test_cases:
            result = extract_symbol(query)
            assert result == expected, f"Expected '{expected}' from '{query}', got '{result}'"


import tempfile
from pathlib import Path

from daytrader.agents.trader import TradingAgent
from daytrader.db.store import DataStore


class TestTradeConfirmation:
    """
    **Feature: day-trading-cli, Property 30: Trade Confirmation Requirement**
    **Validates: Requirements 12.5**
    
    *For any* query that would execute a trade, confirmation should be 
    required before execution.
    """

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=3,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        quantity=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=50)
    def test_buy_requires_confirmation(self, symbol: str, quantity: int):
        """
        *For any* buy trade request, the trading agent should require
        confirmation before executing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            DataStore(db_path)  # Initialize DB
            
            agent = TradingAgent(db_path=db_path)
            agent.set_confirmation_required(True)
            
            # Execute trade without confirmation
            result = agent.execute_trade(
                action="buy",
                symbol=symbol,
                quantity=quantity,
                confirmed=False,
            )
            
            # Should return confirmation request, not execute
            assert "confirm" in result.lower(), (
                f"Buy trade should require confirmation, got: {result}"
            )
            assert symbol.upper() in result, (
                f"Confirmation should mention symbol {symbol}"
            )
            assert str(quantity) in result, (
                f"Confirmation should mention quantity {quantity}"
            )

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=3,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        quantity=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=50)
    def test_sell_requires_confirmation(self, symbol: str, quantity: int):
        """
        *For any* sell trade request, the trading agent should require
        confirmation before executing.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            DataStore(db_path)  # Initialize DB
            
            agent = TradingAgent(db_path=db_path)
            agent.set_confirmation_required(True)
            
            # Execute trade without confirmation
            result = agent.execute_trade(
                action="sell",
                symbol=symbol,
                quantity=quantity,
                confirmed=False,
            )
            
            # Should return confirmation request, not execute
            assert "confirm" in result.lower(), (
                f"Sell trade should require confirmation, got: {result}"
            )

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=3,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        quantity=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=30)
    def test_confirmed_trade_executes(self, symbol: str, quantity: int):
        """
        *For any* trade request with confirmation=True, the trade
        should be executed (or attempted).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            DataStore(db_path)  # Initialize DB
            
            agent = TradingAgent(db_path=db_path)
            agent.set_confirmation_required(True)
            
            # Execute trade with confirmation
            result = agent.execute_trade(
                action="buy",
                symbol=symbol,
                quantity=quantity,
                confirmed=True,
            )
            
            # Should attempt execution (may succeed or fail based on broker)
            # The key is it shouldn't ask for confirmation again
            assert "confirm" not in result.lower() or "executed" in result.lower() or "failed" in result.lower(), (
                f"Confirmed trade should execute or fail, not ask for confirmation again: {result}"
            )

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=3,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        quantity=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=30)
    def test_confirmation_disabled_executes_directly(self, symbol: str, quantity: int):
        """
        *For any* trade request when confirmation is disabled,
        the trade should execute directly.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            DataStore(db_path)  # Initialize DB
            
            agent = TradingAgent(db_path=db_path)
            agent.set_confirmation_required(False)
            
            # Execute trade without confirmation requirement
            result = agent.execute_trade(
                action="buy",
                symbol=symbol,
                quantity=quantity,
                confirmed=False,  # Not confirmed, but confirmation not required
            )
            
            # Should attempt execution directly
            # Result should indicate execution attempt, not confirmation request
            # (It may fail due to paper broker, but shouldn't ask for confirmation)
            is_execution_result = (
                "executed" in result.lower() or 
                "failed" in result.lower() or
                "order" in result.lower() or
                "error" in result.lower()
            )
            is_confirmation_request = "please confirm" in result.lower()
            
            assert is_execution_result or not is_confirmation_request, (
                f"With confirmation disabled, should execute directly: {result}"
            )

    def test_confirmation_includes_trade_details(self):
        """Confirmation request should include all trade details."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            DataStore(db_path)
            
            agent = TradingAgent(db_path=db_path)
            agent.set_confirmation_required(True)
            
            result = agent.execute_trade(
                action="buy",
                symbol="RELIANCE",
                quantity=100,
                order_type="LIMIT",
                product="MIS",
                price=2500.0,
                confirmed=False,
            )
            
            # Check all details are present
            assert "RELIANCE" in result
            assert "100" in result
            assert "LIMIT" in result or "limit" in result.lower()
            assert "MIS" in result
            assert "2500" in result

    def test_default_product_type_is_mis(self):
        """Default product type should be MIS (intraday)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            DataStore(db_path)
            
            agent = TradingAgent(db_path=db_path)
            agent.set_confirmation_required(True)
            
            result = agent.execute_trade(
                action="buy",
                symbol="TCS",
                quantity=50,
                confirmed=False,
            )
            
            # Default product should be MIS
            assert "MIS" in result, "Default product type should be MIS"
