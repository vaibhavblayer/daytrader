"""Property-based tests for technical indicators.

Tests validate indicator calculations against pandas-ta reference implementation.
"""

import math

import pandas as pd
import pandas_ta as ta
from hypothesis import given, settings
from hypothesis import strategies as st

from daytrader.indicators import (
    calculate_rsi,
    calculate_macd,
)


# Strategy for generating realistic price series
@st.composite
def price_series(draw, min_length: int = 50, max_length: int = 200):
    """Generate a realistic price series with positive values and varied movements."""
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    
    # Start with a base price
    base_price = draw(st.floats(min_value=50.0, max_value=500.0))
    
    # Generate percentage changes (realistic daily moves)
    changes = draw(st.lists(
        st.sampled_from([-0.05, -0.04, -0.03, -0.02, -0.01, -0.005,
                         0.005, 0.01, 0.02, 0.03, 0.04, 0.05]),
        min_size=length - 1,
        max_size=length - 1
    ))
    
    prices = [base_price]
    for change in changes:
        new_price = prices[-1] * (1 + change)
        new_price = max(0.01, new_price)
        prices.append(new_price)
    
    return prices


def is_close(a: float, b: float, rel_tolerance: float = 0.02, abs_tolerance: float = 0.5) -> bool:
    """Check if two values are close within tolerance."""
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    
    diff = abs(a - b)
    if diff < abs_tolerance:
        return True
    if max(abs(a), abs(b)) > 0:
        rel_diff = diff / max(abs(a), abs(b))
        if rel_diff < rel_tolerance:
            return True
    return False


class TestRSIAccuracy:
    """
    **Feature: day-trading-cli, Property 9: RSI Calculation Accuracy**
    
    *For any* price series, the calculated RSI should be within acceptable 
    tolerance of reference implementation (pandas-ta).
    
    **Validates: Requirements 4.1, 4.2**
    """
    
    @given(prices=price_series(min_length=50, max_length=200))
    @settings(max_examples=100, deadline=None)
    def test_rsi_matches_pandas_ta(self, prices: list[float]):
        """RSI calculation should match pandas-ta reference within tolerance."""
        period = 14
        
        # Calculate using our implementation
        our_rsi = calculate_rsi(prices, period)
        
        # Calculate using pandas-ta
        df = pd.DataFrame({'close': prices})
        ref_rsi = ta.rsi(df['close'], length=period)
        
        # Skip initial convergence period
        convergence_period = period + 10
        
        matches = 0
        total_compared = 0
        
        for i in range(convergence_period, len(prices)):
            our_val = our_rsi[i]
            ref_val = ref_rsi.iloc[i] if not pd.isna(ref_rsi.iloc[i]) else float('nan')
            
            if math.isnan(our_val) or math.isnan(ref_val):
                continue
                
            total_compared += 1
            if is_close(our_val, ref_val, rel_tolerance=0.05, abs_tolerance=2.0):
                matches += 1
        
        if total_compared > 0:
            match_rate = matches / total_compared
            assert match_rate >= 0.90, (
                f"RSI match rate {match_rate:.2%} below 90% threshold. "
                f"Matched {matches}/{total_compared} values."
            )
        
        # RSI should always be between 0 and 100
        for val in our_rsi:
            if not math.isnan(val):
                assert 0 <= val <= 100, f"RSI value {val} out of range [0, 100]"


class TestMACDAccuracy:
    """
    **Feature: day-trading-cli, Property 10: MACD Calculation Accuracy**
    
    *For any* price series, the calculated MACD, signal, and histogram 
    should match reference implementation (pandas-ta).
    
    **Validates: Requirements 4.1, 4.2**
    """
    
    @given(prices=price_series(min_length=50, max_length=200))
    @settings(max_examples=100, deadline=None)
    def test_macd_matches_pandas_ta(self, prices: list[float]):
        """MACD calculation should match pandas-ta reference within tolerance."""
        fast, slow, signal_period = 12, 26, 9
        
        # Calculate using our implementation
        our_macd, our_signal, our_hist = calculate_macd(prices, fast, slow, signal_period)
        
        # Calculate using pandas-ta
        df = pd.DataFrame({'close': prices})
        ref_macd_df = ta.macd(df['close'], fast=fast, slow=slow, signal=signal_period)
        
        if ref_macd_df is None:
            return
        
        ref_macd = ref_macd_df[f'MACD_{fast}_{slow}_{signal_period}']
        ref_signal = ref_macd_df[f'MACDs_{fast}_{slow}_{signal_period}']
        ref_hist = ref_macd_df[f'MACDh_{fast}_{slow}_{signal_period}']
        
        # Compare MACD line
        macd_matches = 0
        macd_total = 0
        
        for i in range(len(prices)):
            our_val = our_macd[i]
            ref_val = ref_macd.iloc[i] if not pd.isna(ref_macd.iloc[i]) else float('nan')
            
            if math.isnan(our_val) and math.isnan(ref_val):
                continue
            if math.isnan(ref_val):
                continue
                
            macd_total += 1
            if is_close(our_val, ref_val):
                macd_matches += 1
        
        if macd_total > 0:
            match_rate = macd_matches / macd_total
            assert match_rate >= 0.90, (
                f"MACD line match rate {match_rate:.2%} below 90% threshold. "
                f"Matched {macd_matches}/{macd_total} values."
            )
        
        # Compare Signal line
        signal_matches = 0
        signal_total = 0
        
        for i in range(len(prices)):
            our_val = our_signal[i]
            ref_val = ref_signal.iloc[i] if not pd.isna(ref_signal.iloc[i]) else float('nan')
            
            if math.isnan(our_val) and math.isnan(ref_val):
                continue
            if math.isnan(ref_val):
                continue
                
            signal_total += 1
            if is_close(our_val, ref_val):
                signal_matches += 1
        
        if signal_total > 0:
            match_rate = signal_matches / signal_total
            assert match_rate >= 0.90, (
                f"Signal line match rate {match_rate:.2%} below 90% threshold. "
                f"Matched {signal_matches}/{signal_total} values."
            )
        
        # Compare Histogram
        hist_matches = 0
        hist_total = 0
        
        for i in range(len(prices)):
            our_val = our_hist[i]
            ref_val = ref_hist.iloc[i] if not pd.isna(ref_hist.iloc[i]) else float('nan')
            
            if math.isnan(our_val) and math.isnan(ref_val):
                continue
            if math.isnan(ref_val):
                continue
                
            hist_total += 1
            if is_close(our_val, ref_val):
                hist_matches += 1
        
        if hist_total > 0:
            match_rate = hist_matches / hist_total
            assert match_rate >= 0.90, (
                f"Histogram match rate {match_rate:.2%} below 90% threshold. "
                f"Matched {hist_matches}/{hist_total} values."
            )


import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from daytrader.db.store import DataStore
from daytrader.models import Candle
from daytrader.tools.indicators import calculate_indicators


class TestIndicatorDataSource:
    """
    **Feature: day-trading-cli, Property 11: Indicator Data Source**
    **Validates: Requirements 4.3**
    
    *For any* indicator calculation, the data should be sourced from 
    SQLite database.
    """

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=3,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        prices=price_series(min_length=50, max_length=100),
    )
    @settings(max_examples=30, deadline=None)
    def test_indicators_use_database_data(self, symbol: str, prices: list[float]):
        """
        *For any* symbol with data in the database, indicator calculations
        should use that data and return results.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Create candles from price series
            base_date = datetime.now() - timedelta(days=len(prices))
            candles = []
            for i, price in enumerate(prices):
                # Create realistic OHLCV data
                high = price * 1.02
                low = price * 0.98
                candle = Candle(
                    timestamp=base_date + timedelta(days=i),
                    open=price,
                    high=high,
                    low=low,
                    close=price,
                    volume=1000000,
                )
                candles.append(candle)
            
            # Save candles to database
            store.save_candles(symbol, "1day", candles)
            
            # Calculate indicators using the tool (which reads from DB)
            result = calculate_indicators(
                symbol=symbol,
                indicators=["rsi", "macd"],
                timeframe="1day",
                days=len(prices) + 10,
                db_path=db_path,
            )
            
            # Verify data was sourced from database
            assert result["error"] is None, f"Indicator calculation failed: {result['error']}"
            assert result["data_points"] == len(prices), (
                f"Expected {len(prices)} data points, got {result['data_points']}"
            )
            assert result["symbol"] == symbol.upper()
            
            # Verify indicators were calculated
            assert "rsi" in result["indicators"]
            assert "macd" in result["indicators"]

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=3,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
    )
    @settings(max_examples=20)
    def test_indicators_return_error_for_missing_data(self, symbol: str):
        """
        *For any* symbol without data in the database, indicator calculations
        should return an error indicating no data found.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            # Create empty database
            DataStore(db_path)
            
            # Try to calculate indicators for non-existent symbol
            result = calculate_indicators(
                symbol=symbol,
                indicators=["rsi"],
                timeframe="1day",
                days=30,
                db_path=db_path,
            )
            
            # Should return error or empty data
            assert result["data_points"] == 0 or result["error"] is not None

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=3,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        prices=price_series(min_length=50, max_length=100),
    )
    @settings(max_examples=20, deadline=None)
    def test_indicators_match_database_prices(self, symbol: str, prices: list[float]):
        """
        *For any* symbol, the current_price in indicator results should
        match the latest close price in the database.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Create candles from price series
            base_date = datetime.now() - timedelta(days=len(prices))
            candles = []
            for i, price in enumerate(prices):
                candle = Candle(
                    timestamp=base_date + timedelta(days=i),
                    open=price,
                    high=price * 1.02,
                    low=price * 0.98,
                    close=price,
                    volume=1000000,
                )
                candles.append(candle)
            
            # Save candles to database
            store.save_candles(symbol, "1day", candles)
            
            # Calculate indicators
            result = calculate_indicators(
                symbol=symbol,
                indicators=["rsi"],
                timeframe="1day",
                days=len(prices) + 10,
                db_path=db_path,
            )
            
            # Verify current price matches last price in our data
            if result["error"] is None and result["data_points"] > 0:
                expected_price = prices[-1]
                actual_price = result["indicators"].get("current_price")
                assert actual_price is not None, "current_price not in indicators"
                assert abs(actual_price - expected_price) < 0.01, (
                    f"Current price {actual_price} doesn't match expected {expected_price}"
                )
