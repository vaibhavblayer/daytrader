"""Property-based tests for workflow commands.

**Feature: day-trading-cli**
"""

import tempfile
from datetime import date, datetime
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from daytrader.cli.workflow import (
    PREP_REQUIRED_AGENTS,
    calculate_review_metrics,
    get_market_status,
    get_prep_agents,
    is_market_open,
)
from daytrader.db.store import DataStore
from daytrader.models.trade import Trade


class TestPrepAgentInvocation:
    """
    **Feature: day-trading-cli, Property 38: Prep Agent Invocation**
    **Validates: Requirements 16.4**
    
    *For any* prep command, both Research Agent and Data Analyst Agent 
    should be invoked.
    """

    def test_prep_requires_research_and_analyst_agents(self):
        """
        The prep command should always invoke both Research Agent
        and Data Analyst Agent.
        """
        required_agents = get_prep_agents()
        
        assert "research" in required_agents, (
            "Prep should invoke Research Agent"
        )
        assert "analyst" in required_agents, (
            "Prep should invoke Data Analyst Agent"
        )

    def test_prep_required_agents_constant(self):
        """
        PREP_REQUIRED_AGENTS should contain both required agent types.
        """
        assert "research" in PREP_REQUIRED_AGENTS
        assert "analyst" in PREP_REQUIRED_AGENTS
        assert len(PREP_REQUIRED_AGENTS) >= 2

    @given(
        call_count=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=20)
    def test_get_prep_agents_returns_consistent_list(self, call_count: int):
        """
        *For any* number of calls, get_prep_agents should return
        a consistent list containing both required agents.
        """
        for _ in range(call_count):
            agents = get_prep_agents()
            
            assert isinstance(agents, list), "Should return a list"
            assert "research" in agents, "Should include research agent"
            assert "analyst" in agents, "Should include analyst agent"

    def test_get_prep_agents_returns_copy(self):
        """
        get_prep_agents should return a copy, not the original list,
        to prevent accidental modification.
        """
        agents1 = get_prep_agents()
        agents2 = get_prep_agents()
        
        # Modify one list
        agents1.append("test")
        
        # Other list should be unaffected
        assert "test" not in agents2
        assert "test" not in PREP_REQUIRED_AGENTS


class TestMarketStatus:
    """Tests for market status checking functionality."""

    def test_market_status_returns_required_fields(self):
        """Market status should return all required fields."""
        status = get_market_status()
        
        assert "is_open" in status
        assert "message" in status
        assert "current_time" in status
        assert "date" in status
        assert "day" in status

    def test_is_market_open_returns_tuple(self):
        """is_market_open should return a tuple of (bool, str)."""
        result = is_market_open()
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_market_status_is_open_is_boolean(self):
        """is_open field should be a boolean."""
        status = get_market_status()
        assert isinstance(status["is_open"], bool)


class TestReviewMetrics:
    """
    **Feature: day-trading-cli, Property 40: Review Metrics Calculation**
    **Validates: Requirements 17.1, 17.2**
    
    *For any* review command, total P&L, win rate, and average win/loss 
    should be calculated from today's trades.
    """

    def test_empty_trades_returns_zero_metrics(self):
        """Empty trade list should return zero metrics."""
        metrics = calculate_review_metrics([])
        
        assert metrics["total_pnl"] == 0.0
        assert metrics["total_trades"] == 0
        assert metrics["winning_trades"] == 0
        assert metrics["losing_trades"] == 0
        assert metrics["win_rate"] == 0.0
        assert metrics["avg_win"] == 0.0
        assert metrics["avg_loss"] == 0.0

    @given(
        pnl_values=st.lists(
            st.floats(min_value=-10000, max_value=10000, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=100)
    def test_total_pnl_equals_sum_of_trade_pnls(self, pnl_values: list[float]):
        """
        *For any* set of trades, total P&L should equal the sum of 
        individual trade P&Ls.
        """
        trades = []
        for i, pnl in enumerate(pnl_values):
            trade = Trade(
                timestamp=datetime.now(),
                symbol=f"TEST{i}",
                side="BUY",
                quantity=100,
                price=1000.0,
                order_id=f"order_{i}",
                pnl=pnl,
                is_paper=True,
            )
            trades.append(trade)
        
        metrics = calculate_review_metrics(trades)
        expected_total = sum(pnl_values)
        
        assert abs(metrics["total_pnl"] - expected_total) < 0.01, (
            f"Total P&L {metrics['total_pnl']} should equal sum {expected_total}"
        )

    @given(
        winning_count=st.integers(min_value=0, max_value=10),
        losing_count=st.integers(min_value=0, max_value=10),
    )
    @settings(max_examples=100)
    def test_win_rate_calculation(self, winning_count: int, losing_count: int):
        """
        *For any* combination of winning and losing trades, win rate
        should be correctly calculated as (wins / total) * 100.
        """
        trades = []
        
        # Add winning trades
        for i in range(winning_count):
            trade = Trade(
                timestamp=datetime.now(),
                symbol=f"WIN{i}",
                side="BUY",
                quantity=100,
                price=1000.0,
                order_id=f"win_{i}",
                pnl=100.0,  # Positive P&L
                is_paper=True,
            )
            trades.append(trade)
        
        # Add losing trades
        for i in range(losing_count):
            trade = Trade(
                timestamp=datetime.now(),
                symbol=f"LOSE{i}",
                side="BUY",
                quantity=100,
                price=1000.0,
                order_id=f"lose_{i}",
                pnl=-50.0,  # Negative P&L
                is_paper=True,
            )
            trades.append(trade)
        
        metrics = calculate_review_metrics(trades)
        
        total_with_pnl = winning_count + losing_count
        if total_with_pnl > 0:
            expected_win_rate = (winning_count / total_with_pnl) * 100
            assert abs(metrics["win_rate"] - expected_win_rate) < 0.01, (
                f"Win rate {metrics['win_rate']} should be {expected_win_rate}"
            )
        else:
            assert metrics["win_rate"] == 0.0

    @given(
        win_amounts=st.lists(
            st.floats(min_value=1, max_value=10000, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_avg_win_calculation(self, win_amounts: list[float]):
        """
        *For any* set of winning trades, average win should be
        the mean of winning P&Ls.
        """
        trades = []
        for i, amount in enumerate(win_amounts):
            trade = Trade(
                timestamp=datetime.now(),
                symbol=f"WIN{i}",
                side="BUY",
                quantity=100,
                price=1000.0,
                order_id=f"win_{i}",
                pnl=amount,
                is_paper=True,
            )
            trades.append(trade)
        
        metrics = calculate_review_metrics(trades)
        expected_avg = sum(win_amounts) / len(win_amounts)
        
        assert abs(metrics["avg_win"] - expected_avg) < 0.01, (
            f"Avg win {metrics['avg_win']} should be {expected_avg}"
        )

    @given(
        loss_amounts=st.lists(
            st.floats(min_value=1, max_value=10000, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_avg_loss_calculation(self, loss_amounts: list[float]):
        """
        *For any* set of losing trades, average loss should be
        the mean of absolute loss values.
        """
        trades = []
        for i, amount in enumerate(loss_amounts):
            trade = Trade(
                timestamp=datetime.now(),
                symbol=f"LOSE{i}",
                side="BUY",
                quantity=100,
                price=1000.0,
                order_id=f"lose_{i}",
                pnl=-amount,  # Negative P&L
                is_paper=True,
            )
            trades.append(trade)
        
        metrics = calculate_review_metrics(trades)
        expected_avg = sum(loss_amounts) / len(loss_amounts)
        
        assert abs(metrics["avg_loss"] - expected_avg) < 0.01, (
            f"Avg loss {metrics['avg_loss']} should be {expected_avg}"
        )

    @given(
        pnl_values=st.lists(
            st.floats(min_value=-10000, max_value=10000, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=50)
    def test_trade_count_matches_input(self, pnl_values: list[float]):
        """
        *For any* set of trades, total_trades should match the input count.
        """
        trades = []
        for i, pnl in enumerate(pnl_values):
            trade = Trade(
                timestamp=datetime.now(),
                symbol=f"TEST{i}",
                side="BUY",
                quantity=100,
                price=1000.0,
                order_id=f"order_{i}",
                pnl=pnl,
                is_paper=True,
            )
            trades.append(trade)
        
        metrics = calculate_review_metrics(trades)
        
        assert metrics["total_trades"] == len(pnl_values), (
            f"Total trades {metrics['total_trades']} should be {len(pnl_values)}"
        )

    @given(
        pnl_values=st.lists(
            st.floats(min_value=-10000, max_value=10000, allow_nan=False, allow_infinity=False).filter(lambda x: x != 0),
            min_size=2,
            max_size=20,
        ),
    )
    @settings(max_examples=50)
    def test_best_and_worst_trade_identification(self, pnl_values: list[float]):
        """
        *For any* set of trades with P&L, best trade should have highest P&L
        and worst trade should have lowest P&L.
        """
        trades = []
        for i, pnl in enumerate(pnl_values):
            trade = Trade(
                timestamp=datetime.now(),
                symbol=f"TEST{i}",
                side="BUY",
                quantity=100,
                price=1000.0,
                order_id=f"order_{i}",
                pnl=pnl,
                is_paper=True,
            )
            trades.append(trade)
        
        metrics = calculate_review_metrics(trades)
        
        # Find expected best and worst
        positive_pnls = [p for p in pnl_values if p > 0]
        negative_pnls = [p for p in pnl_values if p < 0]
        
        if positive_pnls:
            expected_best = max(positive_pnls)
            assert metrics["best_trade"] is not None
            assert abs(metrics["best_trade"].pnl - expected_best) < 0.01
        
        if negative_pnls:
            expected_worst = min(negative_pnls)
            assert metrics["worst_trade"] is not None
            assert abs(metrics["worst_trade"].pnl - expected_worst) < 0.01

    def test_winning_plus_losing_equals_trades_with_pnl(self):
        """Winning + losing trades should equal trades with non-zero P&L."""
        trades = [
            Trade(
                timestamp=datetime.now(),
                symbol="WIN1",
                side="BUY",
                quantity=100,
                price=1000.0,
                order_id="win_1",
                pnl=100.0,
                is_paper=True,
            ),
            Trade(
                timestamp=datetime.now(),
                symbol="LOSE1",
                side="BUY",
                quantity=100,
                price=1000.0,
                order_id="lose_1",
                pnl=-50.0,
                is_paper=True,
            ),
            Trade(
                timestamp=datetime.now(),
                symbol="ZERO",
                side="BUY",
                quantity=100,
                price=1000.0,
                order_id="zero_1",
                pnl=0.0,  # Zero P&L - neither win nor loss
                is_paper=True,
            ),
        ]
        
        metrics = calculate_review_metrics(trades)
        
        # Zero P&L trades are neither winning nor losing
        assert metrics["winning_trades"] == 1
        assert metrics["losing_trades"] == 1
        assert metrics["total_trades"] == 3
