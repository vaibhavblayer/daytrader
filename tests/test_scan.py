"""Property-based tests for the scan command.

**Feature: day-trading-cli**
"""

import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from daytrader.db.store import DataStore
from daytrader.models import Candle
from daytrader.cli.scan import scan_stocks, get_stock_metrics


def create_test_candles_with_rsi(
    target_rsi: float,
    num_candles: int = 100,
) -> list[Candle]:
    """Create test candles that will produce approximately the target RSI.
    
    Args:
        target_rsi: Target RSI value (0-100).
        num_candles: Number of candles to generate.
        
    Returns:
        List of candles.
    """
    base_date = date.today() - timedelta(days=num_candles)
    candles = []
    
    base_price = 100.0
    
    # Determine trend based on target RSI
    # RSI < 50 means more down days, RSI > 50 means more up days
    up_probability = target_rsi / 100.0
    
    import random
    random.seed(42)  # For reproducibility
    
    for i in range(num_candles):
        candle_date = base_date + timedelta(days=i)
        candle_datetime = datetime.combine(candle_date, datetime.min.time())
        
        # Determine if this is an up or down day
        if random.random() < up_probability:
            # Up day
            change = random.uniform(0.5, 2.0)
        else:
            # Down day
            change = random.uniform(-2.0, -0.5)
        
        price = base_price + change
        base_price = price
        
        candle = Candle(
            timestamp=candle_datetime,
            open=price - change * 0.5,
            high=price + abs(change) * 0.3,
            low=price - abs(change) * 0.3,
            close=price,
            volume=10000 + i * 100,
        )
        candles.append(candle)
    
    return candles


def create_test_candles_with_volume(
    volume_multiplier: float,
    num_candles: int = 50,
) -> list[Candle]:
    """Create test candles with specific volume characteristics.
    
    Args:
        volume_multiplier: Multiplier for the last candle's volume vs average.
        num_candles: Number of candles to generate.
        
    Returns:
        List of candles.
    """
    base_date = date.today() - timedelta(days=num_candles)
    candles = []
    
    base_volume = 10000
    
    for i in range(num_candles):
        candle_date = base_date + timedelta(days=i)
        candle_datetime = datetime.combine(candle_date, datetime.min.time())
        
        # Last candle gets the multiplied volume
        if i == num_candles - 1:
            volume = int(base_volume * volume_multiplier)
        else:
            volume = base_volume
        
        candle = Candle(
            timestamp=candle_datetime,
            open=100.0,
            high=105.0,
            low=97.0,
            close=102.0,
            volume=volume,
        )
        candles.append(candle)
    
    return candles


class TestRSIFilterAccuracy:
    """
    **Feature: day-trading-cli, Property 13: RSI Filter Accuracy**
    **Validates: Requirements 5.1**
    
    *For any* RSI threshold filter, all returned stocks should have RSI
    values satisfying the condition.
    """

    @given(
        rsi_threshold=st.floats(min_value=20.0, max_value=80.0),
        num_symbols=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=50)
    def test_rsi_below_filter_accuracy(
        self, rsi_threshold: float, num_symbols: int
    ):
        """
        *For any* RSI below threshold, all returned stocks should have
        RSI values below that threshold.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Create symbols with varying RSI characteristics
            symbols = []
            for i in range(num_symbols):
                symbol = f"TEST{i}"
                symbols.append(symbol)
                
                # Alternate between low and high RSI targets
                target_rsi = 25.0 if i % 2 == 0 else 75.0
                candles = create_test_candles_with_rsi(target_rsi)
                store.save_candles(symbol, "1day", candles)
                store.add_to_watchlist(symbol, "test")
            
            # Scan with RSI below filter
            results = scan_stocks(
                symbols=symbols,
                store=store,
                rsi_below=rsi_threshold,
            )
            
            # All returned stocks should have RSI below threshold
            for stock in results:
                assert stock["rsi"] is not None, (
                    f"Stock {stock['symbol']} has no RSI value"
                )
                assert stock["rsi"] < rsi_threshold, (
                    f"Stock {stock['symbol']} has RSI {stock['rsi']:.1f} "
                    f"which is not below threshold {rsi_threshold}"
                )

    @given(
        rsi_threshold=st.floats(min_value=20.0, max_value=80.0),
        num_symbols=st.integers(min_value=2, max_value=5),
    )
    @settings(max_examples=50)
    def test_rsi_above_filter_accuracy(
        self, rsi_threshold: float, num_symbols: int
    ):
        """
        *For any* RSI above threshold, all returned stocks should have
        RSI values above that threshold.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Create symbols with varying RSI characteristics
            symbols = []
            for i in range(num_symbols):
                symbol = f"TEST{i}"
                symbols.append(symbol)
                
                # Alternate between low and high RSI targets
                target_rsi = 25.0 if i % 2 == 0 else 75.0
                candles = create_test_candles_with_rsi(target_rsi)
                store.save_candles(symbol, "1day", candles)
            
            # Scan with RSI above filter
            results = scan_stocks(
                symbols=symbols,
                store=store,
                rsi_above=rsi_threshold,
            )
            
            # All returned stocks should have RSI above threshold
            for stock in results:
                assert stock["rsi"] is not None, (
                    f"Stock {stock['symbol']} has no RSI value"
                )
                assert stock["rsi"] > rsi_threshold, (
                    f"Stock {stock['symbol']} has RSI {stock['rsi']:.1f} "
                    f"which is not above threshold {rsi_threshold}"
                )


class TestWatchlistScanScope:
    """
    **Feature: day-trading-cli, Property 16: Watchlist Scan Scope**
    **Validates: Requirements 5.4**
    
    *For any* scan operation, only stocks in the configured watchlist
    should be scanned.
    """

    @given(
        watchlist_symbols=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu",)),
                min_size=2,
                max_size=6,
            ).filter(lambda x: x.strip() != ""),
            min_size=1,
            max_size=5,
            unique=True,
        ),
        non_watchlist_symbols=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu",)),
                min_size=2,
                max_size=6,
            ).filter(lambda x: x.strip() != ""),
            min_size=1,
            max_size=5,
            unique=True,
        ),
    )
    @settings(max_examples=50)
    def test_scan_only_includes_watchlist_symbols(
        self, watchlist_symbols: list[str], non_watchlist_symbols: list[str]
    ):
        """
        *For any* set of watchlist and non-watchlist symbols, scan results
        should only include symbols from the watchlist.
        """
        # Ensure no overlap between watchlist and non-watchlist
        non_watchlist_symbols = [
            s for s in non_watchlist_symbols if s not in watchlist_symbols
        ]
        assume(len(non_watchlist_symbols) > 0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Create data for all symbols
            all_symbols = watchlist_symbols + non_watchlist_symbols
            for symbol in all_symbols:
                candles = create_test_candles_with_rsi(25.0)  # Low RSI to match filter
                store.save_candles(symbol, "1day", candles)
            
            # Scan only watchlist symbols
            results = scan_stocks(
                symbols=watchlist_symbols,  # Only pass watchlist symbols
                store=store,
                rsi_below=50.0,  # Should match all
            )
            
            # All results should be from watchlist
            result_symbols = {stock["symbol"] for stock in results}
            
            for symbol in result_symbols:
                assert symbol in watchlist_symbols, (
                    f"Symbol {symbol} in results but not in watchlist"
                )
            
            # No non-watchlist symbols should appear
            for symbol in non_watchlist_symbols:
                assert symbol not in result_symbols, (
                    f"Non-watchlist symbol {symbol} appeared in results"
                )

    @given(
        num_watchlist=st.integers(min_value=1, max_value=5),
        num_non_watchlist=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=30)
    def test_scan_respects_watchlist_boundary(
        self, num_watchlist: int, num_non_watchlist: int
    ):
        """
        *For any* number of watchlist and non-watchlist stocks, the scan
        should only process watchlist stocks.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Create watchlist symbols
            watchlist_symbols = [f"WATCH{i}" for i in range(num_watchlist)]
            for symbol in watchlist_symbols:
                candles = create_test_candles_with_rsi(25.0)
                store.save_candles(symbol, "1day", candles)
            
            # Create non-watchlist symbols (with data but not in scan list)
            non_watchlist_symbols = [f"OTHER{i}" for i in range(num_non_watchlist)]
            for symbol in non_watchlist_symbols:
                candles = create_test_candles_with_rsi(25.0)
                store.save_candles(symbol, "1day", candles)
            
            # Scan only watchlist
            results = scan_stocks(
                symbols=watchlist_symbols,
                store=store,
                rsi_below=50.0,
            )
            
            # Results should only contain watchlist symbols
            result_symbols = {stock["symbol"] for stock in results}
            assert result_symbols.issubset(set(watchlist_symbols)), (
                f"Results contain non-watchlist symbols: "
                f"{result_symbols - set(watchlist_symbols)}"
            )


class TestVolumeSpikeFilter:
    """Tests for volume spike filtering."""

    @given(
        volume_multiplier=st.floats(min_value=2.5, max_value=10.0),
    )
    @settings(max_examples=30)
    def test_volume_spike_filter_includes_high_volume(
        self, volume_multiplier: float
    ):
        """
        *For any* stock with volume > 2x average, it should be included
        when volume_spike filter is active.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Create stock with high volume
            candles = create_test_candles_with_volume(volume_multiplier)
            store.save_candles("HIGH_VOL", "1day", candles)
            
            # Scan with volume spike filter
            results = scan_stocks(
                symbols=["HIGH_VOL"],
                store=store,
                volume_spike=True,
            )
            
            # Should be included
            assert len(results) == 1, (
                f"Expected 1 result for volume multiplier {volume_multiplier}x"
            )
            assert results[0]["volume_ratio"] >= 2.0

    @given(
        volume_multiplier=st.floats(min_value=0.5, max_value=1.9),
    )
    @settings(max_examples=30)
    def test_volume_spike_filter_excludes_normal_volume(
        self, volume_multiplier: float
    ):
        """
        *For any* stock with volume < 2x average, it should be excluded
        when volume_spike filter is active.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Create stock with normal volume
            candles = create_test_candles_with_volume(volume_multiplier)
            store.save_candles("NORMAL_VOL", "1day", candles)
            
            # Scan with volume spike filter
            results = scan_stocks(
                symbols=["NORMAL_VOL"],
                store=store,
                volume_spike=True,
            )
            
            # Should be excluded
            assert len(results) == 0, (
                f"Expected 0 results for volume multiplier {volume_multiplier}x"
            )


class TestCombinedFilters:
    """Tests for combined filter behavior."""

    def test_combined_filters_require_all_conditions(self):
        """When multiple filters are specified, all must be satisfied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Create stock with low RSI but normal volume
            candles = create_test_candles_with_rsi(25.0)
            store.save_candles("LOW_RSI", "1day", candles)
            
            # Create stock with high volume but normal RSI
            candles = create_test_candles_with_volume(3.0)
            store.save_candles("HIGH_VOL", "1day", candles)
            
            # Scan with both filters - should return nothing
            # (no stock satisfies both conditions)
            results = scan_stocks(
                symbols=["LOW_RSI", "HIGH_VOL"],
                store=store,
                rsi_below=30.0,
                volume_spike=True,
            )
            
            # Each stock only satisfies one condition, not both
            # LOW_RSI has low RSI but normal volume
            # HIGH_VOL has high volume but normal RSI
            for stock in results:
                # If a stock is returned, it must satisfy both conditions
                assert stock["rsi"] < 30.0, "RSI condition not met"
                assert stock["volume_ratio"] >= 2.0, "Volume condition not met"
