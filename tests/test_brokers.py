"""Property-based tests for broker implementations.

**Feature: day-trading-cli**
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from daytrader.brokers.base import Balance, BaseBroker, Quote
from daytrader.brokers.angelone import AngelOneBroker
from daytrader.brokers.paper import PaperBroker
from daytrader.db.store import DataStore
from daytrader.models import Order, OrderResult, Position, Candle


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_smart_api():
    """Create a mock SmartConnect instance."""
    with patch("daytrader.brokers.angelone.SmartConnect") as mock:
        instance = MagicMock()
        mock.return_value = instance
        yield instance


# ============================================================================
# Property 4: Session Token Persistence
# ============================================================================

class TestSessionTokenPersistence:
    """
    **Feature: day-trading-cli, Property 4: Session Token Persistence**
    **Validates: Requirements 2.2**
    
    *For any* successful authentication, the session token should be
    retrievable from storage until explicitly invalidated.
    """

    @given(
        auth_token=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=10,
            max_size=100,
        ).filter(lambda x: x.strip() != ""),
        refresh_token=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=10,
            max_size=100,
        ).filter(lambda x: x.strip() != ""),
        feed_token=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=10,
            max_size=100,
        ).filter(lambda x: x.strip() != ""),
    )
    @settings(max_examples=50)
    def test_session_token_saved_after_login(
        self,
        auth_token: str,
        refresh_token: str,
        feed_token: str,
    ):
        """
        *For any* valid session tokens, after successful login,
        the tokens should be persisted to storage.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "session.json"
            
            with patch("daytrader.brokers.angelone.SmartConnect") as mock_smart:
                mock_instance = MagicMock()
                mock_smart.return_value = mock_instance
                
                # Mock successful login
                mock_instance.generateSession.return_value = {
                    "status": True,
                    "data": {
                        "jwtToken": auth_token,
                        "refreshToken": refresh_token,
                    },
                }
                mock_instance.getfeedToken.return_value = feed_token
                
                broker = AngelOneBroker(
                    api_key="test_api_key",
                    client_id="test_client",
                    pin="1234",
                    totp_secret="JBSWY3DPEHPK3PXP",
                    token_path=token_path,
                )
                
                # Perform login
                result = broker.login()
                
                assert result is True, "Login should succeed"
                assert token_path.exists(), "Session file should be created"
                
                # Verify token is retrievable
                assert broker.get_session_token() == auth_token
                
                # Verify file contents
                session_data = json.loads(token_path.read_text())
                assert session_data["auth_token"] == auth_token
                assert session_data["refresh_token"] == refresh_token
                assert session_data["feed_token"] == feed_token

    @given(
        auth_token=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=10,
            max_size=100,
        ).filter(lambda x: x.strip() != ""),
    )
    @settings(max_examples=50)
    def test_session_token_loadable_after_restart(self, auth_token: str):
        """
        *For any* saved session token, a new broker instance should
        be able to load and use the token.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "session.json"
            
            # Pre-save session data
            session_data = {
                "auth_token": auth_token,
                "refresh_token": "refresh_" + auth_token,
                "feed_token": "feed_" + auth_token,
                "timestamp": datetime.now().isoformat(),
            }
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(json.dumps(session_data))
            
            # Create new broker instance
            broker = AngelOneBroker(
                api_key="test_api_key",
                client_id="test_client",
                pin="1234",
                totp_secret="JBSWY3DPEHPK3PXP",
                token_path=token_path,
            )
            
            # Load session
            loaded = broker._load_session()
            
            assert loaded is True, "Session should load successfully"
            assert broker.get_session_token() == auth_token

    @given(
        auth_token=st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            min_size=10,
            max_size=100,
        ).filter(lambda x: x.strip() != ""),
    )
    @settings(max_examples=50)
    def test_session_token_cleared_after_logout(self, auth_token: str):
        """
        *For any* authenticated session, after logout the token
        should be cleared from storage.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "session.json"
            
            with patch("daytrader.brokers.angelone.SmartConnect") as mock_smart:
                mock_instance = MagicMock()
                mock_smart.return_value = mock_instance
                
                # Mock successful login
                mock_instance.generateSession.return_value = {
                    "status": True,
                    "data": {
                        "jwtToken": auth_token,
                        "refreshToken": "refresh_token",
                    },
                }
                mock_instance.getfeedToken.return_value = "feed_token"
                mock_instance.terminateSession.return_value = {"status": True}
                
                broker = AngelOneBroker(
                    api_key="test_api_key",
                    client_id="test_client",
                    pin="1234",
                    totp_secret="JBSWY3DPEHPK3PXP",
                    token_path=token_path,
                )
                
                # Login then logout
                broker.login()
                assert broker.get_session_token() == auth_token
                
                broker.logout()
                
                # Verify token is cleared
                assert broker.get_session_token() is None
                assert not token_path.exists(), "Session file should be deleted"

    def test_session_load_fails_for_missing_file(self):
        """Session load should fail gracefully when file doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "nonexistent" / "session.json"
            
            broker = AngelOneBroker(
                api_key="test_api_key",
                client_id="test_client",
                pin="1234",
                totp_secret="JBSWY3DPEHPK3PXP",
                token_path=token_path,
            )
            
            loaded = broker._load_session()
            
            assert loaded is False
            assert broker.get_session_token() is None

    def test_session_load_fails_for_invalid_json(self):
        """Session load should fail gracefully for invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "session.json"
            token_path.write_text("not valid json {{{")
            
            broker = AngelOneBroker(
                api_key="test_api_key",
                client_id="test_client",
                pin="1234",
                totp_secret="JBSWY3DPEHPK3PXP",
                token_path=token_path,
            )
            
            loaded = broker._load_session()
            
            assert loaded is False
            assert broker.get_session_token() is None



# ============================================================================
# Property 3: Paper Mode Isolation
# ============================================================================

class TestPaperModeIsolation:
    """
    **Feature: day-trading-cli, Property 3: Paper Mode Isolation**
    **Validates: Requirements 1.5**
    
    *For any* trade executed in paper mode, no external API calls
    to Angel One should be made.
    """

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=2,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        quantity=st.integers(min_value=1, max_value=1000),
        side=st.sampled_from(["BUY", "SELL"]),
    )
    @settings(max_examples=50)
    def test_paper_broker_no_external_calls(
        self,
        symbol: str,
        quantity: int,
        side: str,
    ):
        """
        *For any* order placed through PaperBroker, no external
        API calls should be made.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Create paper broker with sufficient balance
            broker = PaperBroker(
                data_store=store,
                starting_balance=1000000.0,
            )
            broker.login()
            
            # Track all external calls
            with patch("daytrader.brokers.angelone.SmartConnect") as mock_smart:
                # Create order
                order = Order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type="MARKET",
                    product="MIS",
                )
                
                # Execute order
                result = broker.place_order(order)
                
                # Verify no SmartConnect calls were made
                mock_smart.assert_not_called()
                
                # Order should still succeed
                assert result.status in ["COMPLETE", "REJECTED"]

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=2,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
    )
    @settings(max_examples=50)
    def test_paper_broker_quote_no_external_calls(self, symbol: str):
        """
        *For any* quote request through PaperBroker, no external
        API calls should be made.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            broker = PaperBroker(data_store=store)
            broker.login()
            
            with patch("daytrader.brokers.angelone.SmartConnect") as mock_smart:
                quote = broker.get_quote(symbol)
                
                # Verify no SmartConnect calls
                mock_smart.assert_not_called()
                
                # Quote should be returned
                assert quote.symbol == symbol
                assert quote.ltp > 0

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=2,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
    )
    @settings(max_examples=50)
    def test_paper_broker_positions_no_external_calls(self, symbol: str):
        """
        *For any* positions request through PaperBroker, no external
        API calls should be made.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            broker = PaperBroker(
                data_store=store,
                starting_balance=1000000.0,
            )
            broker.login()
            
            # Place an order first to have a position
            order = Order(
                symbol=symbol,
                side="BUY",
                quantity=10,
                order_type="MARKET",
                product="MIS",
            )
            broker.place_order(order)
            
            with patch("daytrader.brokers.angelone.SmartConnect") as mock_smart:
                positions = broker.get_positions()
                
                # Verify no SmartConnect calls
                mock_smart.assert_not_called()
                
                # Should have at least one position
                assert len(positions) >= 1


# ============================================================================
# Property 42: Paper Trading Balance Tracking
# ============================================================================

class TestPaperTradingBalance:
    """
    **Feature: day-trading-cli, Property 42: Paper Trading Balance Tracking**
    **Validates: Requirements 18.2**
    
    *For any* paper trade, the virtual balance should be updated correctly
    (decreased on buy, increased on sell).
    """

    @given(
        quantity=st.integers(min_value=1, max_value=100),
        price=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_balance_decreases_on_buy(self, quantity: int, price: float):
        """
        *For any* buy order, the balance should decrease by approximately
        the order value (accounting for slippage).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            starting_balance = 1000000.0
            broker = PaperBroker(
                data_store=store,
                starting_balance=starting_balance,
                slippage_percent=0.0,  # No slippage for predictable testing
            )
            broker.login()
            
            # Save a candle so we have a known price
            from daytrader.models import Candle
            candle = Candle(
                timestamp=datetime.now(),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=10000,
            )
            store.save_candles("TEST", "1day", [candle])
            
            initial_balance = broker.get_current_balance()
            
            order = Order(
                symbol="TEST",
                side="BUY",
                quantity=quantity,
                order_type="MARKET",
                product="MIS",
            )
            
            result = broker.place_order(order)
            
            if result.status == "COMPLETE":
                final_balance = broker.get_current_balance()
                expected_cost = result.filled_price * quantity
                
                # Balance should decrease by order value
                assert final_balance < initial_balance
                assert abs((initial_balance - final_balance) - expected_cost) < 0.01

    @given(
        buy_quantity=st.integers(min_value=10, max_value=100),
        sell_quantity=st.integers(min_value=1, max_value=10),
        price=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_balance_increases_on_sell(self, buy_quantity: int, sell_quantity: int, price: float):
        """
        *For any* sell order, the balance should increase by approximately
        the order value (accounting for slippage).
        """
        # Ensure sell quantity is less than buy quantity
        assume(sell_quantity <= buy_quantity)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            broker = PaperBroker(
                data_store=store,
                starting_balance=1000000.0,
                slippage_percent=0.0,
            )
            broker.login()
            
            # Save a candle for known price
            from daytrader.models import Candle
            candle = Candle(
                timestamp=datetime.now(),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=10000,
            )
            store.save_candles("TEST", "1day", [candle])
            
            # First buy to have a position
            buy_order = Order(
                symbol="TEST",
                side="BUY",
                quantity=buy_quantity,
                order_type="MARKET",
                product="MIS",
            )
            broker.place_order(buy_order)
            
            balance_before_sell = broker.get_current_balance()
            
            # Now sell
            sell_order = Order(
                symbol="TEST",
                side="SELL",
                quantity=sell_quantity,
                order_type="MARKET",
                product="MIS",
            )
            
            result = broker.place_order(sell_order)
            
            if result.status == "COMPLETE":
                final_balance = broker.get_current_balance()
                expected_proceeds = result.filled_price * sell_quantity
                
                # Balance should increase by order value
                assert final_balance > balance_before_sell
                assert abs((final_balance - balance_before_sell) - expected_proceeds) < 0.01

    @given(
        starting_balance=st.floats(min_value=10000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_reset_restores_starting_balance(self, starting_balance: float):
        """
        *For any* starting balance, reset should restore the balance
        to the starting amount.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            broker = PaperBroker(
                data_store=store,
                starting_balance=starting_balance,
            )
            broker.login()
            
            # Make some trades
            order = Order(
                symbol="TEST",
                side="BUY",
                quantity=10,
                order_type="MARKET",
                product="MIS",
            )
            broker.place_order(order)
            
            # Balance should have changed
            current = broker.get_current_balance()
            assert current != starting_balance
            
            # Reset
            broker.reset()
            
            # Balance should be restored
            assert broker.get_current_balance() == starting_balance
            
            # Positions should be cleared
            positions = broker.get_positions()
            assert len(positions) == 0

    @given(
        quantity=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_insufficient_balance_rejected(self, quantity: int):
        """
        *For any* order that exceeds available balance, the order
        should be rejected.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            # Very small starting balance
            broker = PaperBroker(
                data_store=store,
                starting_balance=100.0,
            )
            broker.login()
            
            # Save a candle with high price
            from daytrader.models import Candle
            candle = Candle(
                timestamp=datetime.now(),
                open=1000.0,
                high=1000.0,
                low=1000.0,
                close=1000.0,
                volume=10000,
            )
            store.save_candles("EXPENSIVE", "1day", [candle])
            
            order = Order(
                symbol="EXPENSIVE",
                side="BUY",
                quantity=quantity,
                order_type="MARKET",
                product="MIS",
            )
            
            result = broker.place_order(order)
            
            # Order should be rejected due to insufficient funds
            assert result.status == "REJECTED"
            assert "Insufficient" in result.message

    @given(
        num_trades=st.integers(min_value=1, max_value=5),
        quantities=st.lists(
            st.integers(min_value=1, max_value=10),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=30)
    def test_balance_consistency_across_trades(self, num_trades: int, quantities: list[int]):
        """
        *For any* sequence of trades, the balance should remain consistent
        with the sum of all transactions.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            starting_balance = 1000000.0
            broker = PaperBroker(
                data_store=store,
                starting_balance=starting_balance,
                slippage_percent=0.0,
            )
            broker.login()
            
            # Save a candle
            from daytrader.models import Candle
            price = 100.0
            candle = Candle(
                timestamp=datetime.now(),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=10000,
            )
            store.save_candles("TEST", "1day", [candle])
            
            total_spent = 0.0
            
            for qty in quantities[:num_trades]:
                order = Order(
                    symbol="TEST",
                    side="BUY",
                    quantity=qty,
                    order_type="MARKET",
                    product="MIS",
                )
                
                result = broker.place_order(order)
                if result.status == "COMPLETE":
                    total_spent += result.filled_price * qty
            
            final_balance = broker.get_current_balance()
            
            # Balance should equal starting minus total spent
            assert abs(final_balance - (starting_balance - total_spent)) < 0.01


# ============================================================================
# Property 18: Default Product Type
# ============================================================================

class TestDefaultProductType:
    """
    **Feature: day-trading-cli, Property 18: Default Product Type**
    **Validates: Requirements 6.7**
    
    *For any* order without explicit product flag, the product type
    should default to MIS.
    """

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=2,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        quantity=st.integers(min_value=1, max_value=100),
        side=st.sampled_from(["BUY", "SELL"]),
        order_type=st.sampled_from(["MARKET", "LIMIT"]),
        price=st.floats(min_value=10.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_order_defaults_to_mis(
        self,
        symbol: str,
        quantity: int,
        side: str,
        order_type: str,
        price: float,
    ):
        """
        *For any* order created without specifying product type,
        the product should default to MIS.
        """
        # Create order without specifying product (uses default)
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price if order_type == "LIMIT" else None,
        )
        
        # Verify default product is MIS
        assert order.product == "MIS", f"Expected product to be MIS, got {order.product}"

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=2,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        quantity=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100)
    def test_order_with_explicit_cnc(self, symbol: str, quantity: int):
        """
        *For any* order with explicit CNC product type,
        the product should be CNC.
        """
        order = Order(
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            order_type="MARKET",
            product="CNC",
        )
        
        assert order.product == "CNC", f"Expected product to be CNC, got {order.product}"

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=2,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        quantity=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100)
    def test_order_with_explicit_mis(self, symbol: str, quantity: int):
        """
        *For any* order with explicit MIS product type,
        the product should be MIS.
        """
        order = Order(
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            order_type="MARKET",
            product="MIS",
        )
        
        assert order.product == "MIS", f"Expected product to be MIS, got {order.product}"

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=2,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        quantity=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=50)
    def test_paper_broker_respects_default_product(self, symbol: str, quantity: int):
        """
        *For any* order placed through PaperBroker without explicit product,
        the position should have MIS product type.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            broker = PaperBroker(
                data_store=store,
                starting_balance=1000000.0,
                slippage_percent=0.0,
            )
            broker.login()
            
            # Save a candle for known price
            candle = Candle(
                timestamp=datetime.now(),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=10000,
            )
            store.save_candles(symbol, "1day", [candle])
            
            # Create order without specifying product (uses default MIS)
            order = Order(
                symbol=symbol,
                side="BUY",
                quantity=quantity,
                order_type="MARKET",
            )
            
            result = broker.place_order(order)
            
            if result.status == "COMPLETE":
                positions = broker.get_positions()
                matching_pos = [p for p in positions if p.symbol == symbol]
                
                assert len(matching_pos) > 0, f"Expected position for {symbol}"
                assert matching_pos[0].product == "MIS", f"Expected MIS product, got {matching_pos[0].product}"


# ============================================================================
# Property 20: Position Exit Completeness
# ============================================================================

class TestPositionExitCompleteness:
    """
    **Feature: day-trading-cli, Property 20: Position Exit Completeness**
    **Validates: Requirements 7.2**
    
    *For any* exit command for a symbol, the entire position should be closed.
    """

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=2,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        quantity=st.integers(min_value=1, max_value=100),
    )
    @settings(max_examples=100)
    def test_exit_closes_entire_position(self, symbol: str, quantity: int):
        """
        *For any* position, when exiting by symbol, the entire position
        quantity should be sold and the position should be removed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            broker = PaperBroker(
                data_store=store,
                starting_balance=1000000.0,
                slippage_percent=0.0,
            )
            broker.login()
            
            # Save a candle for known price
            candle = Candle(
                timestamp=datetime.now(),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=10000,
            )
            store.save_candles(symbol, "1day", [candle])
            
            # Create a position by buying
            buy_order = Order(
                symbol=symbol,
                side="BUY",
                quantity=quantity,
                order_type="MARKET",
                product="MIS",
            )
            
            buy_result = broker.place_order(buy_order)
            assert buy_result.status == "COMPLETE", "Buy order should complete"
            
            # Verify position exists
            positions_before = broker.get_positions()
            matching_before = [p for p in positions_before if p.symbol == symbol]
            assert len(matching_before) == 1, f"Expected 1 position for {symbol}"
            assert matching_before[0].quantity == quantity, "Position quantity should match"
            
            # Exit the position by selling the entire quantity
            sell_order = Order(
                symbol=symbol,
                side="SELL",
                quantity=quantity,
                order_type="MARKET",
                product="MIS",
            )
            
            sell_result = broker.place_order(sell_order)
            assert sell_result.status == "COMPLETE", "Sell order should complete"
            assert sell_result.filled_qty == quantity, "Should sell entire quantity"
            
            # Verify position is closed
            positions_after = broker.get_positions()
            matching_after = [p for p in positions_after if p.symbol == symbol]
            assert len(matching_after) == 0, f"Position for {symbol} should be closed"

    @given(
        symbols=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu",)),
                min_size=2,
                max_size=6,
            ).filter(lambda x: x.strip() != ""),
            min_size=2,
            max_size=4,
            unique=True,
        ),
        quantities=st.lists(
            st.integers(min_value=1, max_value=50),
            min_size=2,
            max_size=4,
        ),
    )
    @settings(max_examples=50)
    def test_exit_single_symbol_preserves_others(self, symbols: list[str], quantities: list[int]):
        """
        *For any* set of positions, exiting one symbol should not affect
        positions in other symbols.
        """
        # Ensure we have matching lengths
        assume(len(symbols) >= 2)
        assume(len(quantities) >= len(symbols))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            broker = PaperBroker(
                data_store=store,
                starting_balance=10000000.0,
                slippage_percent=0.0,
            )
            broker.login()
            
            # Create positions for all symbols
            for i, symbol in enumerate(symbols):
                candle = Candle(
                    timestamp=datetime.now(),
                    open=100.0,
                    high=100.0,
                    low=100.0,
                    close=100.0,
                    volume=10000,
                )
                store.save_candles(symbol, "1day", [candle])
                
                order = Order(
                    symbol=symbol,
                    side="BUY",
                    quantity=quantities[i],
                    order_type="MARKET",
                    product="MIS",
                )
                broker.place_order(order)
            
            # Verify all positions exist
            positions_before = broker.get_positions()
            assert len(positions_before) == len(symbols), "All positions should exist"
            
            # Exit only the first symbol
            symbol_to_exit = symbols[0]
            qty_to_exit = quantities[0]
            
            sell_order = Order(
                symbol=symbol_to_exit,
                side="SELL",
                quantity=qty_to_exit,
                order_type="MARKET",
                product="MIS",
            )
            broker.place_order(sell_order)
            
            # Verify only the exited position is gone
            positions_after = broker.get_positions()
            remaining_symbols = [p.symbol for p in positions_after]
            
            assert symbol_to_exit not in remaining_symbols, f"{symbol_to_exit} should be closed"
            
            # All other symbols should still have positions
            for symbol in symbols[1:]:
                assert symbol in remaining_symbols, f"{symbol} should still have position"

    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=("Lu",)),
            min_size=2,
            max_size=10,
        ).filter(lambda x: x.strip() != ""),
        initial_qty=st.integers(min_value=10, max_value=100),
        partial_qty=st.integers(min_value=1, max_value=9),
    )
    @settings(max_examples=50)
    def test_partial_exit_reduces_position(self, symbol: str, initial_qty: int, partial_qty: int):
        """
        *For any* partial exit (selling less than full position),
        the remaining position should reflect the correct quantity.
        """
        assume(partial_qty < initial_qty)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            broker = PaperBroker(
                data_store=store,
                starting_balance=1000000.0,
                slippage_percent=0.0,
            )
            broker.login()
            
            # Save a candle
            candle = Candle(
                timestamp=datetime.now(),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=10000,
            )
            store.save_candles(symbol, "1day", [candle])
            
            # Create initial position
            buy_order = Order(
                symbol=symbol,
                side="BUY",
                quantity=initial_qty,
                order_type="MARKET",
                product="MIS",
            )
            broker.place_order(buy_order)
            
            # Partial exit
            sell_order = Order(
                symbol=symbol,
                side="SELL",
                quantity=partial_qty,
                order_type="MARKET",
                product="MIS",
            )
            broker.place_order(sell_order)
            
            # Verify remaining position
            positions = broker.get_positions()
            matching = [p for p in positions if p.symbol == symbol]
            
            assert len(matching) == 1, f"Position for {symbol} should still exist"
            expected_remaining = initial_qty - partial_qty
            assert matching[0].quantity == expected_remaining, \
                f"Expected {expected_remaining} remaining, got {matching[0].quantity}"


# ============================================================================
# Property 45: Paper Trading Reset
# ============================================================================

class TestPaperTradingReset:
    """
    **Feature: day-trading-cli, Property 45: Paper Trading Reset**
    **Validates: Requirements 18.5**
    
    *For any* paper reset command, the balance should be reset to the
    configured starting amount and all paper positions cleared.
    """

    @given(
        starting_balance=st.floats(
            min_value=10000.0,
            max_value=10000000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        num_trades=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=100)
    def test_reset_restores_starting_balance(
        self,
        starting_balance: float,
        num_trades: int,
    ):
        """
        *For any* starting balance configuration, after reset the balance
        should equal the configured starting amount.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            broker = PaperBroker(
                data_store=store,
                starting_balance=starting_balance,
                slippage_percent=0.0,
            )
            broker.login()
            
            # Save a candle for trading
            candle = Candle(
                timestamp=datetime.now(),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=10000,
            )
            store.save_candles("TEST", "1day", [candle])
            
            # Make some trades to change balance
            for i in range(num_trades):
                order = Order(
                    symbol="TEST",
                    side="BUY",
                    quantity=10,
                    order_type="MARKET",
                    product="MIS",
                )
                broker.place_order(order)
            
            # Verify balance has changed
            balance_before_reset = broker.get_current_balance()
            assert balance_before_reset != starting_balance, \
                "Balance should have changed after trades"
            
            # Reset
            broker.reset()
            
            # Verify balance is restored
            balance_after_reset = broker.get_current_balance()
            assert abs(balance_after_reset - starting_balance) < 0.01, \
                f"Expected balance {starting_balance}, got {balance_after_reset}"

    @given(
        symbols=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Lu",)),
                min_size=2,
                max_size=6,
            ).filter(lambda x: x.strip() != ""),
            min_size=1,
            max_size=5,
            unique=True,
        ),
        quantities=st.lists(
            st.integers(min_value=1, max_value=50),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=100)
    def test_reset_clears_all_positions(
        self,
        symbols: list[str],
        quantities: list[int],
    ):
        """
        *For any* set of open positions, after reset all positions
        should be cleared.
        """
        assume(len(quantities) >= len(symbols))
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            broker = PaperBroker(
                data_store=store,
                starting_balance=10000000.0,
                slippage_percent=0.0,
            )
            broker.login()
            
            # Create positions for all symbols
            for i, symbol in enumerate(symbols):
                candle = Candle(
                    timestamp=datetime.now(),
                    open=100.0,
                    high=100.0,
                    low=100.0,
                    close=100.0,
                    volume=10000,
                )
                store.save_candles(symbol, "1day", [candle])
                
                order = Order(
                    symbol=symbol,
                    side="BUY",
                    quantity=quantities[i],
                    order_type="MARKET",
                    product="MIS",
                )
                broker.place_order(order)
            
            # Verify positions exist
            positions_before = broker.get_positions()
            assert len(positions_before) == len(symbols), \
                f"Expected {len(symbols)} positions, got {len(positions_before)}"
            
            # Reset
            broker.reset()
            
            # Verify all positions are cleared
            positions_after = broker.get_positions()
            assert len(positions_after) == 0, \
                f"Expected 0 positions after reset, got {len(positions_after)}"

    @given(
        starting_balance=st.floats(
            min_value=10000.0,
            max_value=1000000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=100)
    def test_reset_is_idempotent(self, starting_balance: float):
        """
        *For any* paper trading state, calling reset multiple times
        should produce the same result as calling it once.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            broker = PaperBroker(
                data_store=store,
                starting_balance=starting_balance,
                slippage_percent=0.0,
            )
            broker.login()
            
            # Make some trades
            candle = Candle(
                timestamp=datetime.now(),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=10000,
            )
            store.save_candles("TEST", "1day", [candle])
            
            order = Order(
                symbol="TEST",
                side="BUY",
                quantity=10,
                order_type="MARKET",
                product="MIS",
            )
            broker.place_order(order)
            
            # Reset once
            broker.reset()
            balance_after_first_reset = broker.get_current_balance()
            positions_after_first_reset = len(broker.get_positions())
            
            # Reset again
            broker.reset()
            balance_after_second_reset = broker.get_current_balance()
            positions_after_second_reset = len(broker.get_positions())
            
            # Results should be identical
            assert abs(balance_after_first_reset - balance_after_second_reset) < 0.01
            assert positions_after_first_reset == positions_after_second_reset == 0
            assert abs(balance_after_second_reset - starting_balance) < 0.01

    @given(
        starting_balance=st.floats(
            min_value=10000.0,
            max_value=1000000.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=100)
    def test_reset_allows_new_trades(self, starting_balance: float):
        """
        *For any* reset operation, the broker should be able to
        accept new trades immediately after reset.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = DataStore(db_path)
            
            broker = PaperBroker(
                data_store=store,
                starting_balance=starting_balance,
                slippage_percent=0.0,
            )
            broker.login()
            
            # Save candle
            candle = Candle(
                timestamp=datetime.now(),
                open=100.0,
                high=100.0,
                low=100.0,
                close=100.0,
                volume=10000,
            )
            store.save_candles("TEST", "1day", [candle])
            
            # Make initial trade
            order1 = Order(
                symbol="TEST",
                side="BUY",
                quantity=10,
                order_type="MARKET",
                product="MIS",
            )
            broker.place_order(order1)
            
            # Reset
            broker.reset()
            
            # Make new trade after reset
            order2 = Order(
                symbol="TEST",
                side="BUY",
                quantity=5,
                order_type="MARKET",
                product="MIS",
            )
            result = broker.place_order(order2)
            
            # New trade should succeed
            assert result.status == "COMPLETE", "Trade after reset should succeed"
            
            # Position should reflect only the new trade
            positions = broker.get_positions()
            assert len(positions) == 1
            assert positions[0].quantity == 5
