"""Property-based tests for portfolio functionality.

**Feature: day-trading-cli**
"""

import tempfile
from datetime import datetime, date
from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from daytrader.cli.portfolio import calculate_pnl_from_trades
from daytrader.db.store import DataStore
from daytrader.models import Trade


# Strategy for generating valid trade data
def trade_strategy():
    """Generate valid Trade objects for testing."""
    return st.builds(
        Trade,
        id=st.none(),
        timestamp=st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2025, 12, 31),
        ),
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=1,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        side=st.sampled_from(["BUY", "SELL"]),
        quantity=st.integers(min_value=1, max_value=10000),
        price=st.floats(min_value=0.01, max_value=100000.0, allow_nan=False, allow_infinity=False),
        order_id=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Nd")),
            min_size=1,
            max_size=20,
        ).filter(lambda x: x.strip() != ""),
        pnl=st.one_of(
            st.none(),
            st.floats(min_value=-100000.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
        ),
        is_paper=st.booleans(),
    )


class TestPnLCalculationAccuracy:
    """
    **Feature: day-trading-cli, Property 23: P&L Calculation Accuracy**
    **Validates: Requirements 8.1**
    
    *For any* set of trades, the calculated P&L should equal sum of
    individual trade P&Ls.
    """

    @given(
        trades=st.lists(trade_strategy(), min_size=0, max_size=50)
    )
    @settings(max_examples=100)
    def test_pnl_sum_equals_total(self, trades: list[Trade]):
        """
        *For any* list of trades, the total realized P&L should equal
        the sum of all individual trade P&Ls.
        """
        result = calculate_pnl_from_trades(trades)
        
        # Calculate expected P&L by summing individual trade P&Ls
        expected_pnl = sum(t.pnl for t in trades if t.pnl is not None)
        
        # The calculated realized_pnl should match the sum
        assert abs(result["realized_pnl"] - expected_pnl) < 0.01, (
            f"P&L mismatch: calculated {result['realized_pnl']}, expected {expected_pnl}"
        )

    @given(
        trades=st.lists(trade_strategy(), min_size=0, max_size=50)
    )
    @settings(max_examples=100)
    def test_trade_count_accuracy(self, trades: list[Trade]):
        """
        *For any* list of trades, the total_trades count should equal
        the number of trades in the input.
        """
        result = calculate_pnl_from_trades(trades)
        
        assert result["total_trades"] == len(trades), (
            f"Trade count mismatch: calculated {result['total_trades']}, expected {len(trades)}"
        )

    @given(
        trades=st.lists(trade_strategy(), min_size=0, max_size=50)
    )
    @settings(max_examples=100)
    def test_winning_losing_trade_counts(self, trades: list[Trade]):
        """
        *For any* list of trades, the sum of winning and losing trades
        should equal the number of trades with non-null P&L.
        """
        result = calculate_pnl_from_trades(trades)
        
        # Count trades with P&L
        trades_with_pnl = sum(1 for t in trades if t.pnl is not None)
        winning = sum(1 for t in trades if t.pnl is not None and t.pnl > 0)
        losing = sum(1 for t in trades if t.pnl is not None and t.pnl < 0)
        
        assert result["winning_trades"] == winning, (
            f"Winning trades mismatch: calculated {result['winning_trades']}, expected {winning}"
        )
        assert result["losing_trades"] == losing, (
            f"Losing trades mismatch: calculated {result['losing_trades']}, expected {losing}"
        )

    @given(
        trades=st.lists(trade_strategy(), min_size=1, max_size=50)
    )
    @settings(max_examples=100)
    def test_win_rate_bounds(self, trades: list[Trade]):
        """
        *For any* list of trades, the win rate should be between 0 and 100.
        """
        result = calculate_pnl_from_trades(trades)
        
        assert 0 <= result["win_rate"] <= 100, (
            f"Win rate out of bounds: {result['win_rate']}"
        )

    @given(
        trades=st.lists(trade_strategy(), min_size=1, max_size=50)
    )
    @settings(max_examples=100)
    def test_win_rate_calculation(self, trades: list[Trade]):
        """
        *For any* list of trades with P&L, win rate should equal
        (winning_trades / trades_with_pnl) * 100.
        """
        result = calculate_pnl_from_trades(trades)
        
        trades_with_pnl = result["winning_trades"] + result["losing_trades"]
        
        if trades_with_pnl > 0:
            expected_win_rate = (result["winning_trades"] / trades_with_pnl) * 100
            assert abs(result["win_rate"] - expected_win_rate) < 0.01, (
                f"Win rate mismatch: calculated {result['win_rate']}, expected {expected_win_rate}"
            )
        else:
            assert result["win_rate"] == 0.0, "Win rate should be 0 when no trades have P&L"

    def test_empty_trades_returns_zeros(self):
        """Empty trade list should return all zeros."""
        result = calculate_pnl_from_trades([])
        
        assert result["realized_pnl"] == 0.0
        assert result["total_trades"] == 0
        assert result["winning_trades"] == 0
        assert result["losing_trades"] == 0
        assert result["win_rate"] == 0.0
        assert result["avg_win"] == 0.0
        assert result["avg_loss"] == 0.0

    @given(
        winning_pnls=st.lists(
            st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=50)
    def test_avg_win_calculation(self, winning_pnls: list[float]):
        """
        *For any* set of winning trades, avg_win should equal
        sum(winning_pnls) / count(winning_trades).
        """
        trades = [
            Trade(
                timestamp=datetime.now(),
                symbol="TEST",
                side="SELL",
                quantity=1,
                price=100.0,
                order_id=f"ORD{i}",
                pnl=pnl,
                is_paper=True,
            )
            for i, pnl in enumerate(winning_pnls)
        ]
        
        result = calculate_pnl_from_trades(trades)
        
        expected_avg_win = sum(winning_pnls) / len(winning_pnls)
        assert abs(result["avg_win"] - expected_avg_win) < 0.01, (
            f"Avg win mismatch: calculated {result['avg_win']}, expected {expected_avg_win}"
        )

    @given(
        losing_pnls=st.lists(
            st.floats(min_value=-10000.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=50)
    def test_avg_loss_calculation(self, losing_pnls: list[float]):
        """
        *For any* set of losing trades, avg_loss should equal
        sum(abs(losing_pnls)) / count(losing_trades).
        """
        trades = [
            Trade(
                timestamp=datetime.now(),
                symbol="TEST",
                side="SELL",
                quantity=1,
                price=100.0,
                order_id=f"ORD{i}",
                pnl=pnl,
                is_paper=True,
            )
            for i, pnl in enumerate(losing_pnls)
        ]
        
        result = calculate_pnl_from_trades(trades)
        
        expected_avg_loss = sum(abs(pnl) for pnl in losing_pnls) / len(losing_pnls)
        assert abs(result["avg_loss"] - expected_avg_loss) < 0.01, (
            f"Avg loss mismatch: calculated {result['avg_loss']}, expected {expected_avg_loss}"
        )


class TestTradeLoggingCompleteness:
    """
    **Feature: day-trading-cli, Property 22: Trade Logging Completeness**
    **Validates: Requirements 8.4**
    
    *For any* executed trade, a corresponding entry should exist in the trades table.
    """

    @given(
        trade=trade_strategy()
    )
    @settings(max_examples=100)
    def test_trade_logged_and_retrievable(self, trade: Trade):
        """
        *For any* valid trade, logging it should make it retrievable
        from the database.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Log the trade
            store.log_trade(trade)
            
            # Retrieve all trades
            trades = store.get_trades()
            
            # Verify the trade exists
            assert len(trades) >= 1, "No trades found after logging"
            
            # Find the logged trade
            found = False
            for t in trades:
                if (t.symbol == trade.symbol and 
                    t.side == trade.side and 
                    t.quantity == trade.quantity and
                    abs(t.price - trade.price) < 0.01 and
                    t.order_id == trade.order_id):
                    found = True
                    break
            
            assert found, f"Logged trade not found in database: {trade}"

    @given(
        trades=st.lists(trade_strategy(), min_size=1, max_size=20, unique_by=lambda t: t.order_id)
    )
    @settings(max_examples=50)
    def test_multiple_trades_logged(self, trades: list[Trade]):
        """
        *For any* list of trades with unique order IDs, all should be
        retrievable after logging.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Log all trades
            for trade in trades:
                store.log_trade(trade)
            
            # Retrieve all trades
            retrieved = store.get_trades()
            
            # Verify count matches
            assert len(retrieved) == len(trades), (
                f"Trade count mismatch: logged {len(trades)}, retrieved {len(retrieved)}"
            )
            
            # Verify all order IDs are present
            retrieved_order_ids = {t.order_id for t in retrieved}
            for trade in trades:
                assert trade.order_id in retrieved_order_ids, (
                    f"Trade with order_id {trade.order_id} not found"
                )

    @given(
        trade=trade_strategy()
    )
    @settings(max_examples=50)
    def test_trade_fields_preserved(self, trade: Trade):
        """
        *For any* logged trade, all fields should be preserved correctly.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Log the trade
            store.log_trade(trade)
            
            # Retrieve trades
            trades = store.get_trades()
            assert len(trades) == 1, "Expected exactly one trade"
            
            retrieved = trades[0]
            
            # Verify all fields
            assert retrieved.symbol == trade.symbol, "Symbol mismatch"
            assert retrieved.side == trade.side, "Side mismatch"
            assert retrieved.quantity == trade.quantity, "Quantity mismatch"
            assert abs(retrieved.price - trade.price) < 0.01, "Price mismatch"
            assert retrieved.order_id == trade.order_id, "Order ID mismatch"
            assert retrieved.is_paper == trade.is_paper, "is_paper mismatch"
            
            # P&L comparison (handle None)
            if trade.pnl is None:
                assert retrieved.pnl is None, "P&L should be None"
            else:
                assert retrieved.pnl is not None, "P&L should not be None"
                assert abs(retrieved.pnl - trade.pnl) < 0.01, "P&L mismatch"

    @given(
        trade=trade_strategy(),
        target_date=st.dates(min_value=date(2020, 1, 1), max_value=date(2025, 12, 31))
    )
    @settings(max_examples=50)
    def test_trade_date_filter(self, trade: Trade, target_date: date):
        """
        *For any* trade, filtering by date should return it only if
        the trade timestamp matches the filter date.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Log the trade
            store.log_trade(trade)
            
            # Retrieve trades for the target date
            trades = store.get_trades(trade_date=target_date)
            
            # Check if trade should be in results
            trade_date = trade.timestamp.date()
            
            if trade_date == target_date:
                assert len(trades) >= 1, "Trade should be found for matching date"
                found = any(t.order_id == trade.order_id for t in trades)
                assert found, "Trade not found for matching date"
            else:
                # Trade should not be in results for different date
                found = any(t.order_id == trade.order_id for t in trades)
                assert not found, "Trade should not be found for different date"
