"""Property-based tests for data caching functionality.

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


class TestCachePersistence:
    """
    **Feature: day-trading-cli, Property 7: Cache Persistence**
    **Validates: Requirements 3.2, 3.3**
    
    *For any* fetched data, querying the same data immediately after should
    return results from cache without API calls.
    """

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=1,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        timeframe=st.sampled_from(["1min", "5min", "15min", "1hour", "1day"]),
        num_candles=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100)
    def test_candle_cache_round_trip(
        self, symbol: str, timeframe: str, num_candles: int
    ):
        """
        *For any* set of candles saved to the database, querying the same
        symbol and timeframe should return the exact same candles.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)

            # Generate candles with valid data
            base_date = date.today() - timedelta(days=num_candles)
            candles = []
            
            for i in range(num_candles):
                candle_date = base_date + timedelta(days=i)
                candle_datetime = datetime.combine(candle_date, datetime.min.time())
                
                # Generate realistic OHLCV data
                base_price = 100.0 + i
                candle = Candle(
                    timestamp=candle_datetime,
                    open=base_price,
                    high=base_price + 5.0,
                    low=base_price - 3.0,
                    close=base_price + 2.0,
                    volume=10000 + i * 100,
                )
                candles.append(candle)

            # Save candles to cache
            store.save_candles(symbol, timeframe, candles)

            # Query the same data back
            from_date = base_date
            to_date = date.today()
            cached_candles = store.get_candles(symbol, timeframe, from_date, to_date)

            # Verify all candles are retrieved
            assert len(cached_candles) == len(candles), (
                f"Expected {len(candles)} candles, got {len(cached_candles)}"
            )

            # Verify candle data matches
            for original, cached in zip(candles, cached_candles):
                assert original.timestamp == cached.timestamp
                assert original.open == cached.open
                assert original.high == cached.high
                assert original.low == cached.low
                assert original.close == cached.close
                assert original.volume == cached.volume

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=1,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        timeframe=st.sampled_from(["1min", "5min", "15min", "1hour", "1day"]),
    )
    @settings(max_examples=50)
    def test_cache_returns_empty_for_missing_data(self, symbol: str, timeframe: str):
        """
        *For any* symbol with no cached data, querying should return an empty list.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)

            # Query for data that doesn't exist
            from_date = date.today() - timedelta(days=30)
            to_date = date.today()
            cached_candles = store.get_candles(symbol, timeframe, from_date, to_date)

            assert cached_candles == [], "Expected empty list for missing data"

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=1,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        timeframes=st.lists(
            st.sampled_from(["1min", "5min", "15min", "1hour", "1day"]),
            min_size=2,
            max_size=3,
            unique=True,
        ),
    )
    @settings(max_examples=50)
    def test_cache_separates_timeframes(self, symbol: str, timeframes: list[str]):
        """
        *For any* symbol, data cached for different timeframes should be
        stored and retrieved independently.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)

            base_date = date.today() - timedelta(days=5)
            
            # Save different candles for each timeframe
            for i, timeframe in enumerate(timeframes):
                candles = []
                for j in range(3):
                    candle_date = base_date + timedelta(days=j)
                    candle_datetime = datetime.combine(candle_date, datetime.min.time())
                    
                    # Use different prices for each timeframe to distinguish them
                    base_price = 100.0 + (i * 50)
                    candle = Candle(
                        timestamp=candle_datetime,
                        open=base_price,
                        high=base_price + 5.0,
                        low=base_price - 3.0,
                        close=base_price + 2.0,
                        volume=10000,
                    )
                    candles.append(candle)
                
                store.save_candles(symbol, timeframe, candles)

            # Verify each timeframe has its own data
            from_date = base_date
            to_date = date.today()
            
            for i, timeframe in enumerate(timeframes):
                cached = store.get_candles(symbol, timeframe, from_date, to_date)
                expected_base_price = 100.0 + (i * 50)
                
                assert len(cached) == 3, f"Expected 3 candles for {timeframe}"
                for candle in cached:
                    assert candle.open == expected_base_price, (
                        f"Wrong data for timeframe {timeframe}"
                    )

    @given(
        symbols=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu",)),
                min_size=1,
                max_size=10,
            ).filter(lambda x: x.strip() != ""),
            min_size=2,
            max_size=4,
            unique=True,
        ),
        timeframe=st.sampled_from(["1day"]),
    )
    @settings(max_examples=50)
    def test_cache_separates_symbols(self, symbols: list[str], timeframe: str):
        """
        *For any* set of symbols, data cached for each should be stored
        and retrieved independently.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)

            base_date = date.today() - timedelta(days=5)
            
            # Save different candles for each symbol
            for i, symbol in enumerate(symbols):
                candles = []
                for j in range(3):
                    candle_date = base_date + timedelta(days=j)
                    candle_datetime = datetime.combine(candle_date, datetime.min.time())
                    
                    # Use different prices for each symbol
                    base_price = 100.0 + (i * 100)
                    candle = Candle(
                        timestamp=candle_datetime,
                        open=base_price,
                        high=base_price + 5.0,
                        low=base_price - 3.0,
                        close=base_price + 2.0,
                        volume=10000,
                    )
                    candles.append(candle)
                
                store.save_candles(symbol, timeframe, candles)

            # Verify each symbol has its own data
            from_date = base_date
            to_date = date.today()
            
            for i, symbol in enumerate(symbols):
                cached = store.get_candles(symbol, timeframe, from_date, to_date)
                expected_base_price = 100.0 + (i * 100)
                
                assert len(cached) == 3, f"Expected 3 candles for {symbol}"
                for candle in cached:
                    assert candle.open == expected_base_price, (
                        f"Wrong data for symbol {symbol}"
                    )

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=1,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        timeframe=st.sampled_from(["1day"]),
        num_candles=st.integers(min_value=5, max_value=20),
    )
    @settings(max_examples=50)
    def test_cache_date_range_filtering(
        self, symbol: str, timeframe: str, num_candles: int
    ):
        """
        *For any* cached data, querying with a date range should return
        only candles within that range.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)

            base_date = date.today() - timedelta(days=num_candles)
            candles = []
            
            for i in range(num_candles):
                candle_date = base_date + timedelta(days=i)
                candle_datetime = datetime.combine(candle_date, datetime.min.time())
                
                candle = Candle(
                    timestamp=candle_datetime,
                    open=100.0,
                    high=105.0,
                    low=97.0,
                    close=102.0,
                    volume=10000,
                )
                candles.append(candle)

            store.save_candles(symbol, timeframe, candles)

            # Query a subset of the date range
            mid_point = num_candles // 2
            query_from = base_date + timedelta(days=mid_point)
            query_to = date.today()
            
            cached = store.get_candles(symbol, timeframe, query_from, query_to)
            
            # All returned candles should be within the query range
            for candle in cached:
                candle_date = candle.timestamp.date()
                assert candle_date >= query_from, (
                    f"Candle date {candle_date} is before query start {query_from}"
                )
                assert candle_date <= query_to, (
                    f"Candle date {candle_date} is after query end {query_to}"
                )

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=1,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        timeframe=st.sampled_from(["1day"]),
    )
    @settings(max_examples=50)
    def test_cache_update_replaces_existing(self, symbol: str, timeframe: str):
        """
        *For any* cached candle, saving a new candle with the same timestamp
        should replace the existing one (upsert behavior).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)

            candle_date = date.today() - timedelta(days=1)
            candle_datetime = datetime.combine(candle_date, datetime.min.time())
            
            # Save initial candle
            initial_candle = Candle(
                timestamp=candle_datetime,
                open=100.0,
                high=105.0,
                low=97.0,
                close=102.0,
                volume=10000,
            )
            store.save_candles(symbol, timeframe, [initial_candle])

            # Save updated candle with same timestamp
            updated_candle = Candle(
                timestamp=candle_datetime,
                open=110.0,
                high=115.0,
                low=107.0,
                close=112.0,
                volume=20000,
            )
            store.save_candles(symbol, timeframe, [updated_candle])

            # Query should return only one candle with updated values
            from_date = candle_date
            to_date = date.today()
            cached = store.get_candles(symbol, timeframe, from_date, to_date)
            
            assert len(cached) == 1, "Expected exactly one candle after update"
            assert cached[0].open == 110.0, "Candle was not updated"
            assert cached[0].volume == 20000, "Candle volume was not updated"
