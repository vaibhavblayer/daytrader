"""Property-based tests for the analyze command.

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
from daytrader.cli.analyze import (
    calculate_indicators_for_symbol,
    _parse_indicators,
    AVAILABLE_INDICATORS,
    DEFAULT_INDICATORS,
)


def create_test_candles(num_candles: int = 100) -> list[Candle]:
    """Create test candles with realistic price data."""
    base_date = date.today() - timedelta(days=num_candles)
    candles = []
    
    base_price = 100.0
    for i in range(num_candles):
        candle_date = base_date + timedelta(days=i)
        candle_datetime = datetime.combine(candle_date, datetime.min.time())
        
        # Add some variation to make realistic data
        variation = (i % 10) - 5
        price = base_price + variation
        
        candle = Candle(
            timestamp=candle_datetime,
            open=price,
            high=price + 3.0,
            low=price - 2.0,
            close=price + 1.0,
            volume=10000 + i * 100,
        )
        candles.append(candle)
    
    return candles


class TestSelectiveIndicatorCalculation:
    """
    **Feature: day-trading-cli, Property 12: Selective Indicator Calculation**
    **Validates: Requirements 4.4**
    
    *For any* indicator filter specification, only the specified indicators
    should be calculated.
    """

    @given(
        selected_indicators=st.lists(
            st.sampled_from(AVAILABLE_INDICATORS),
            min_size=1,
            max_size=len(AVAILABLE_INDICATORS),
            unique=True,
        )
    )
    @settings(max_examples=100)
    def test_only_requested_indicators_calculated(
        self, selected_indicators: list[str]
    ):
        """
        *For any* subset of available indicators, only those indicators
        should appear in the results.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Create and save test candles
            candles = create_test_candles(100)
            store.save_candles("TEST", "1day", candles)
            
            # Calculate indicators
            results = calculate_indicators_for_symbol(
                "TEST", selected_indicators, store
            )
            
            # Verify no error
            assert "error" not in results, f"Unexpected error: {results.get('error')}"
            
            # Get calculated indicators
            calculated = set(results.get("indicators", {}).keys())
            requested = set(selected_indicators)
            
            # All calculated indicators should be in the requested set
            for ind in calculated:
                assert ind in requested, (
                    f"Indicator '{ind}' was calculated but not requested. "
                    f"Requested: {requested}, Calculated: {calculated}"
                )

    @given(
        excluded_indicators=st.lists(
            st.sampled_from(AVAILABLE_INDICATORS),
            min_size=1,
            max_size=len(AVAILABLE_INDICATORS) - 1,
            unique=True,
        )
    )
    @settings(max_examples=100)
    def test_excluded_indicators_not_calculated(
        self, excluded_indicators: list[str]
    ):
        """
        *For any* set of excluded indicators, those indicators should not
        appear in the results when requesting only the complement set.
        """
        # Get the complement set (indicators NOT excluded)
        selected = [ind for ind in AVAILABLE_INDICATORS if ind not in excluded_indicators]
        
        # Skip if no indicators would be selected
        assume(len(selected) > 0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Create and save test candles
            candles = create_test_candles(100)
            store.save_candles("TEST", "1day", candles)
            
            # Calculate only selected indicators
            results = calculate_indicators_for_symbol("TEST", selected, store)
            
            # Verify no error
            assert "error" not in results, f"Unexpected error: {results.get('error')}"
            
            # Get calculated indicators
            calculated = set(results.get("indicators", {}).keys())
            
            # None of the excluded indicators should be calculated
            for ind in excluded_indicators:
                assert ind not in calculated, (
                    f"Excluded indicator '{ind}' was calculated. "
                    f"Selected: {selected}, Calculated: {calculated}"
                )

    @given(
        indicator=st.sampled_from(AVAILABLE_INDICATORS)
    )
    @settings(max_examples=50)
    def test_single_indicator_calculation(self, indicator: str):
        """
        *For any* single indicator, requesting only that indicator should
        result in only that indicator being calculated.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Create and save test candles
            candles = create_test_candles(100)
            store.save_candles("TEST", "1day", candles)
            
            # Calculate single indicator
            results = calculate_indicators_for_symbol("TEST", [indicator], store)
            
            # Verify no error
            assert "error" not in results, f"Unexpected error: {results.get('error')}"
            
            # Get calculated indicators
            calculated = list(results.get("indicators", {}).keys())
            
            # Should have at most one indicator (the requested one)
            assert len(calculated) <= 1, (
                f"Expected at most 1 indicator, got {len(calculated)}: {calculated}"
            )
            
            # If calculated, it should be the requested one
            if calculated:
                assert calculated[0] == indicator, (
                    f"Expected '{indicator}', got '{calculated[0]}'"
                )


class TestIndicatorParsing:
    """Tests for indicator string parsing."""

    @given(
        indicators=st.lists(
            st.sampled_from(AVAILABLE_INDICATORS),
            min_size=1,
            max_size=len(AVAILABLE_INDICATORS),
            unique=True,
        )
    )
    @settings(max_examples=50)
    def test_parse_valid_indicators(self, indicators: list[str]):
        """
        *For any* list of valid indicators, parsing the comma-separated
        string should return the same indicators.
        """
        indicator_str = ",".join(indicators)
        parsed = _parse_indicators(indicator_str)
        
        # All requested indicators should be in parsed result
        for ind in indicators:
            assert ind in parsed, f"Indicator '{ind}' not found in parsed result"

    @given(
        valid_indicators=st.lists(
            st.sampled_from(AVAILABLE_INDICATORS),
            min_size=1,
            max_size=3,
            unique=True,
        ),
        invalid_indicators=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Ll",)),
                min_size=3,
                max_size=10,
            ).filter(lambda x: x not in AVAILABLE_INDICATORS),
            min_size=1,
            max_size=3,
        ),
    )
    @settings(max_examples=50)
    def test_parse_filters_invalid_indicators(
        self, valid_indicators: list[str], invalid_indicators: list[str]
    ):
        """
        *For any* mix of valid and invalid indicators, parsing should
        return only the valid ones.
        """
        all_indicators = valid_indicators + invalid_indicators
        indicator_str = ",".join(all_indicators)
        parsed = _parse_indicators(indicator_str)
        
        # All valid indicators should be present
        for ind in valid_indicators:
            assert ind in parsed, f"Valid indicator '{ind}' not found"
        
        # No invalid indicators should be present
        for ind in invalid_indicators:
            assert ind not in parsed, f"Invalid indicator '{ind}' should not be present"

    def test_parse_empty_returns_defaults(self):
        """Parsing empty or invalid string should return defaults."""
        # Empty string
        parsed = _parse_indicators("")
        assert parsed == DEFAULT_INDICATORS
        
        # All invalid
        parsed = _parse_indicators("invalid1,invalid2")
        assert parsed == DEFAULT_INDICATORS

    @given(
        indicators=st.lists(
            st.sampled_from(AVAILABLE_INDICATORS),
            min_size=1,
            max_size=len(AVAILABLE_INDICATORS),
            unique=True,
        ),
        extra_spaces=st.booleans(),
    )
    @settings(max_examples=50)
    def test_parse_handles_whitespace(
        self, indicators: list[str], extra_spaces: bool
    ):
        """
        *For any* indicator list, parsing should handle extra whitespace.
        """
        if extra_spaces:
            indicator_str = " , ".join(indicators)
        else:
            indicator_str = ",".join(indicators)
        
        parsed = _parse_indicators(indicator_str)
        
        # All indicators should be found regardless of whitespace
        for ind in indicators:
            assert ind in parsed, f"Indicator '{ind}' not found with whitespace"


class TestIndicatorDataIntegrity:
    """Tests for indicator calculation data integrity."""

    @given(
        num_candles=st.integers(min_value=50, max_value=100),
    )
    @settings(max_examples=30)
    def test_indicators_use_all_available_data(self, num_candles: int):
        """
        *For any* amount of cached data within the query range, indicator 
        calculations should use the available data and report the candle count.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Create and save test candles
            candles = create_test_candles(num_candles)
            store.save_candles("TEST", "1day", candles)
            
            # Calculate all indicators with matching days parameter
            results = calculate_indicators_for_symbol(
                "TEST", AVAILABLE_INDICATORS, store, days=num_candles
            )
            
            # Verify no error
            assert "error" not in results, f"Unexpected error: {results.get('error')}"
            
            # Verify candle count is reported
            assert "candle_count" in results
            assert results["candle_count"] == num_candles

    def test_insufficient_data_returns_error(self):
        """With insufficient data, an error should be returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Create only 10 candles (insufficient)
            candles = create_test_candles(10)
            store.save_candles("TEST", "1day", candles)
            
            # Calculate indicators
            results = calculate_indicators_for_symbol(
                "TEST", AVAILABLE_INDICATORS, store
            )
            
            # Should return error
            assert "error" in results
            assert "Insufficient data" in results["error"]
