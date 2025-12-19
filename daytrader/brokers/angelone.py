"""Angel One broker implementation using SmartAPI."""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import pyotp
from SmartApi import SmartConnect

from daytrader.brokers.base import Balance, BaseBroker, Quote
from daytrader.models import Candle, Order, OrderResult, Position


# Interval mapping for SmartAPI
INTERVAL_MAP = {
    "1min": "ONE_MINUTE",
    "5min": "FIVE_MINUTE",
    "15min": "FIFTEEN_MINUTE",
    "1hour": "ONE_HOUR",
    "1day": "ONE_DAY",
}

# Exchange mapping
EXCHANGE_MAP = {
    "NSE": "NSE",
    "BSE": "BSE",
    "NFO": "NFO",
}


class AngelOneBroker(BaseBroker):
    """Angel One broker implementation using SmartAPI.
    
    Handles authentication with TOTP, session management,
    and all trading operations through Angel One's SmartAPI.
    """

    def __init__(
        self,
        api_key: str,
        client_id: str,
        pin: str,
        totp_secret: str,
        token_path: Optional[Path] = None,
    ):
        """Initialize Angel One broker.
        
        Args:
            api_key: Angel One API key.
            client_id: Angel One client ID.
            pin: Angel One PIN.
            totp_secret: TOTP secret for 2FA.
            token_path: Path to store session tokens.
        """
        self.api_key = api_key
        self.client_id = client_id
        self.pin = pin
        self.totp_secret = totp_secret
        self.token_path = token_path or Path.home() / ".config" / "daytrader" / "session.json"
        
        self._smart_api: Optional[SmartConnect] = None
        self._auth_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._feed_token: Optional[str] = None

    def _generate_totp(self) -> str:
        """Generate TOTP code for authentication."""
        # Clean the TOTP secret - remove common separators and convert to uppercase
        clean_secret = self.totp_secret.replace("-", "").replace(" ", "").replace("_", "").upper()
        
        # Validate base32 characters (A-Z, 2-7)
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
        invalid_chars = set(clean_secret) - valid_chars
        if invalid_chars:
            raise ValueError(f"TOTP secret contains invalid characters: {invalid_chars}. "
                           f"Base32 only allows A-Z and 2-7. Your secret: {clean_secret[:4]}...")
        
        totp = pyotp.TOTP(clean_secret)
        return totp.now()

    def _save_session(self) -> None:
        """Save session tokens to file."""
        if not self._auth_token:
            return
            
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        session_data = {
            "auth_token": self._auth_token,
            "refresh_token": self._refresh_token,
            "feed_token": self._feed_token,
            "timestamp": datetime.now().isoformat(),
        }
        self.token_path.write_text(json.dumps(session_data))

    def _load_session(self) -> bool:
        """Load session tokens from file.
        
        Returns:
            True if valid session loaded, False otherwise.
        """
        if not self.token_path.exists():
            return False
            
        try:
            session_data = json.loads(self.token_path.read_text())
            self._auth_token = session_data.get("auth_token")
            self._refresh_token = session_data.get("refresh_token")
            self._feed_token = session_data.get("feed_token")
            return bool(self._auth_token)
        except (json.JSONDecodeError, KeyError):
            return False

    def _clear_session(self) -> None:
        """Clear stored session tokens."""
        self._auth_token = None
        self._refresh_token = None
        self._feed_token = None
        if self.token_path.exists():
            self.token_path.unlink()

    def get_session_token(self) -> Optional[str]:
        """Get the current session token.
        
        Returns:
            Session token if authenticated, None otherwise.
        """
        return self._auth_token

    def login(self) -> bool:
        """Authenticate with Angel One using TOTP.
        
        Returns:
            True if authentication successful, False otherwise.
            
        Raises:
            RuntimeError: If authentication fails with details.
        """
        try:
            self._smart_api = SmartConnect(api_key=self.api_key)
            totp_code = self._generate_totp()
            
            data = self._smart_api.generateSession(
                clientCode=self.client_id,
                password=self.pin,
                totp=totp_code,
            )
            
            if data and data.get("status"):
                self._auth_token = data["data"]["jwtToken"]
                self._refresh_token = data["data"]["refreshToken"]
                self._feed_token = self._smart_api.getfeedToken()
                self._save_session()
                return True
            
            # Get error message from response
            error_msg = data.get("message", "Unknown error") if data else "No response from API"
            self._last_error = error_msg
            return False
        except Exception as e:
            self._last_error = str(e)
            return False
    
    def get_last_error(self) -> str:
        """Get the last error message from login attempt."""
        return getattr(self, "_last_error", "Unknown error")

    def logout(self) -> bool:
        """Logout and invalidate session.
        
        Returns:
            True if logout successful, False otherwise.
        """
        try:
            if self._smart_api and self._auth_token:
                self._smart_api.terminateSession(self.client_id)
            self._clear_session()
            self._smart_api = None
            return True
        except Exception:
            self._clear_session()
            return False

    def is_authenticated(self) -> bool:
        """Check if currently authenticated.
        
        Returns:
            True if authenticated, False otherwise.
        """
        return self._auth_token is not None

    def _ensure_authenticated(self) -> None:
        """Ensure we have a valid session, re-authenticating if needed."""
        if not self.is_authenticated():
            # Try to load existing session
            if self._load_session():
                # Recreate SmartConnect with loaded token
                self._smart_api = SmartConnect(api_key=self.api_key)
                # Remove "Bearer " prefix if present (SmartAPI adds it)
                token = self._auth_token
                if token and token.startswith("Bearer "):
                    token = token[7:]
                self._smart_api.setAccessToken(token)
                if self._refresh_token:
                    self._smart_api.setRefreshToken(self._refresh_token)
            else:
                # Need to login fresh
                if not self.login():
                    raise RuntimeError("Failed to authenticate with Angel One")
        
        # If SmartConnect not initialized, do it now
        if self._smart_api is None:
            self._smart_api = SmartConnect(api_key=self.api_key)
            token = self._auth_token
            if token and token.startswith("Bearer "):
                token = token[7:]
            self._smart_api.setAccessToken(token)
            if self._refresh_token:
                self._smart_api.setRefreshToken(self._refresh_token)
                # Generate new token from refresh token (as per official docs)
                try:
                    self._smart_api.generateToken(self._refresh_token)
                except Exception:
                    pass  # Ignore if token refresh fails

    def _get_symbol_info(self, symbol: str, exchange: str = "NSE") -> tuple[str, str]:
        """Get the token and proper trading symbol for a symbol.
        
        Args:
            symbol: Trading symbol (user input like YESBANK).
            exchange: Exchange (NSE, BSE, NFO).
            
        Returns:
            Tuple of (symbol_token, trading_symbol) for API calls.
        """
        self._ensure_authenticated()
        
        # Common NSE symbol tokens (hardcoded for popular stocks)
        # These don't need -EQ suffix
        common_tokens = {
            "RELIANCE": ("2885", "RELIANCE-EQ"),
            "TCS": ("11536", "TCS-EQ"),
            "INFY": ("1594", "INFY-EQ"),
            "HDFCBANK": ("1333", "HDFCBANK-EQ"),
            "ICICIBANK": ("4963", "ICICIBANK-EQ"),
            "SBIN": ("3045", "SBIN-EQ"),
            "BHARTIARTL": ("10604", "BHARTIARTL-EQ"),
            "ITC": ("1660", "ITC-EQ"),
            "KOTAKBANK": ("1922", "KOTAKBANK-EQ"),
            "LT": ("11483", "LT-EQ"),
            "AXISBANK": ("5900", "AXISBANK-EQ"),
            "HINDUNILVR": ("1394", "HINDUNILVR-EQ"),
            "BAJFINANCE": ("317", "BAJFINANCE-EQ"),
            "MARUTI": ("10999", "MARUTI-EQ"),
            "ASIANPAINT": ("236", "ASIANPAINT-EQ"),
            "TITAN": ("3506", "TITAN-EQ"),
            "WIPRO": ("3787", "WIPRO-EQ"),
            "HCLTECH": ("7229", "HCLTECH-EQ"),
            "SUNPHARMA": ("3351", "SUNPHARMA-EQ"),
            "TATAMOTORS": ("3456", "TATAMOTORS-EQ"),
            "TATASTEEL": ("3499", "TATASTEEL-EQ"),
            "POWERGRID": ("14977", "POWERGRID-EQ"),
            "NTPC": ("11630", "NTPC-EQ"),
            "ONGC": ("2475", "ONGC-EQ"),
            "COALINDIA": ("20374", "COALINDIA-EQ"),
            "NIFTY": ("99926000", "NIFTY"),
            "BANKNIFTY": ("99926009", "BANKNIFTY"),
        }
        
        symbol_upper = symbol.upper()
        if symbol_upper in common_tokens:
            return common_tokens[symbol_upper]
        
        # Try to search for the symbol using API
        try:
            search_result = self._smart_api.searchScrip(exchange, symbol_upper)
            if search_result and search_result.get("data"):
                # First try exact match
                for item in search_result["data"]:
                    if item.get("tradingsymbol", "").upper() == symbol_upper:
                        return (item.get("symboltoken", symbol), item.get("tradingsymbol", symbol))
                
                # Then try with -EQ suffix (equity segment - most common for stocks)
                for item in search_result["data"]:
                    trading_symbol = item.get("tradingsymbol", "")
                    if trading_symbol.upper() == f"{symbol_upper}-EQ":
                        return (item.get("symboltoken", ""), trading_symbol)
                
                # Try any result ending with -EQ (equity segment)
                for item in search_result["data"]:
                    trading_symbol = item.get("tradingsymbol", "")
                    if trading_symbol.upper().endswith("-EQ"):
                        return (item.get("symboltoken", ""), trading_symbol)
                
                # Finally, take the first result if available
                if search_result["data"]:
                    first = search_result["data"][0]
                    return (first.get("symboltoken", ""), first.get("tradingsymbol", symbol))
        except Exception:
            pass
        
        # Fallback to symbol name (may not work for all API calls)
        return (symbol, symbol)

    def _get_symbol_token(self, symbol: str, exchange: str = "NSE") -> str:
        """Get the token for a symbol (backward compatibility).
        
        Args:
            symbol: Trading symbol.
            exchange: Exchange (NSE, BSE, NFO).
            
        Returns:
            Symbol token for API calls.
        """
        token, _ = self._get_symbol_info(symbol, exchange)
        return token

    def get_quote(self, symbol: str) -> Quote:
        """Get real-time quote for a symbol.
        
        Args:
            symbol: Trading symbol.
            
        Returns:
            Quote with current market data.
        """
        self._ensure_authenticated()
        
        try:
            # Get proper symbol token and trading symbol
            symbol_token, trading_symbol = self._get_symbol_info(symbol)
            
            ltp_data = self._smart_api.ltpData(
                exchange="NSE",
                tradingsymbol=trading_symbol,
                symboltoken=symbol_token,
            )
            
            if not ltp_data or not ltp_data.get("data"):
                raise ValueError(f"Failed to get quote for {symbol}")
            
            data = ltp_data["data"]
            ltp = float(data.get("ltp", 0))
            close = float(data.get("close", ltp))
            change = ltp - close
            change_percent = (change / close * 100) if close > 0 else 0
            
            return Quote(
                symbol=symbol,
                ltp=ltp,
                change=change,
                change_percent=change_percent,
                volume=int(data.get("volume", 0)),
                open=float(data.get("open", ltp)),
                high=float(data.get("high", ltp)),
                low=float(data.get("low", ltp)),
                close=close,
            )
        except Exception as e:
            raise ValueError(f"Failed to get quote for {symbol}: {e}")

    def get_historical(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        interval: str,
    ) -> list[Candle]:
        """Get historical OHLCV data.
        
        Args:
            symbol: Trading symbol.
            from_date: Start date.
            to_date: End date.
            interval: Candle interval.
            
        Returns:
            List of candles.
        """
        self._ensure_authenticated()
        
        if interval not in INTERVAL_MAP:
            raise ValueError(f"Invalid interval: {interval}. Must be one of {list(INTERVAL_MAP.keys())}")
        
        try:
            historic_params = {
                "exchange": "NSE",
                "symboltoken": self._get_symbol_token(symbol),
                "interval": INTERVAL_MAP[interval],
                "fromdate": f"{from_date.isoformat()} 09:15",
                "todate": f"{to_date.isoformat()} 15:30",
            }
            
            data = self._smart_api.getCandleData(historic_params)
            
            if not data or not data.get("data"):
                return []
            
            candles = []
            for row in data["data"]:
                candles.append(Candle(
                    timestamp=datetime.fromisoformat(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=int(row[5]),
                ))
            
            return candles
        except Exception as e:
            raise ValueError(f"Failed to get historical data for {symbol}: {e}")

    def place_order(self, order: Order) -> OrderResult:
        """Place an order via Angel One.
        
        Args:
            order: Order to place.
            
        Returns:
            OrderResult with execution details.
        """
        self._ensure_authenticated()
        
        try:
            # Use variety from order (NORMAL or AMO)
            variety = getattr(order, 'variety', 'NORMAL') or 'NORMAL'
            
            # Get proper symbol token and trading symbol
            symbol_token, trading_symbol = self._get_symbol_info(order.symbol)
            
            # Map product types to SmartAPI values
            product_map = {
                "MIS": "INTRADAY",
                "CNC": "DELIVERY",
                "NRML": "CARRYFORWARD",
                "INTRADAY": "INTRADAY",
                "DELIVERY": "DELIVERY",
            }
            product_type = product_map.get(order.product.upper(), "INTRADAY")
            
            order_params = {
                "variety": variety,
                "tradingsymbol": trading_symbol,
                "symboltoken": symbol_token,
                "transactiontype": order.side,
                "exchange": "NSE",
                "ordertype": order.order_type,
                "producttype": product_type,
                "duration": "DAY",
                "quantity": str(order.quantity),
                "price": str(order.price) if order.price else "0",
                "triggerprice": str(order.trigger_price) if order.trigger_price else "0",
            }
            
            # Log order params for debugging
            import logging
            logging.info(f"Placing order with params: {order_params}")
            
            try:
                response = self._smart_api.placeOrder(order_params)
            except Exception as api_error:
                error_msg = str(api_error)
                # Check if it's a JSON parsing error with empty response
                if "Couldn't parse" in error_msg and "b''" in error_msg:
                    return OrderResult(
                        order_id="",
                        status="ERROR",
                        filled_qty=0,
                        filled_price=0.0,
                        message="Broker API returned empty response. Try re-login with 'daytrader login'",
                    )
                # Return the actual error message from broker
                return OrderResult(
                    order_id="",
                    status="ERROR",
                    filled_qty=0,
                    filled_price=0.0,
                    message=error_msg,
                )
            
            logging.info(f"Order response: {response}")
            
            if response is None:
                return OrderResult(
                    order_id="",
                    status="ERROR",
                    filled_qty=0,
                    filled_price=0.0,
                    message="Empty response from broker API",
                )
            
            # Handle string response (order ID directly returned)
            if isinstance(response, str):
                return OrderResult(
                    order_id=response,
                    status="PLACED",
                    filled_qty=0,
                    filled_price=0.0,
                    message="Order placed successfully",
                )
            
            if response.get("status"):
                order_id = (
                    response.get("data", {}).get("orderid", "")
                    if response.get("data")
                    else ""
                )
                return OrderResult(
                    order_id=order_id,
                    status="PLACED",
                    filled_qty=0,
                    filled_price=0.0,
                    message="Order placed successfully",
                )
            else:
                return OrderResult(
                    order_id="",
                    status="REJECTED",
                    filled_qty=0,
                    filled_price=0.0,
                    message=response.get("message", "Order placement failed"),
                )
        except Exception as e:
            return OrderResult(
                order_id="",
                status="ERROR",
                filled_qty=0,
                filled_price=0.0,
                message=str(e),
            )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order.
        
        Args:
            order_id: ID of the order to cancel.
            
        Returns:
            True if cancellation successful.
        """
        self._ensure_authenticated()
        
        try:
            response = self._smart_api.cancelOrder(
                order_id=order_id,
                variety="NORMAL",
            )
            return response and response.get("status", False)
        except Exception:
            return False

    def get_positions(self) -> list[Position]:
        """Get all open positions.
        
        Returns:
            List of current positions.
        """
        self._ensure_authenticated()
        
        try:
            response = self._smart_api.position()
            
            if not response or not response.get("data"):
                return []
            
            positions = []
            for pos in response["data"]:
                qty = int(pos.get("netqty", 0))
                if qty == 0:
                    continue
                
                # Use the correct field names from Angel One API
                # For buy positions: buyavgprice or totalbuyavgprice
                # For sell positions: sellavgprice
                # avgnetprice is the net average price
                avg_price = float(pos.get("avgnetprice", 0) or pos.get("buyavgprice", 0) or 0)
                ltp = float(pos.get("ltp", 0))
                
                # Use the pre-calculated P&L from API (unrealised for open positions)
                pnl = float(pos.get("unrealised", 0) or pos.get("pnl", 0) or 0)
                
                # Calculate P&L percentage
                cost = avg_price * abs(qty)
                pnl_percent = (pnl / cost * 100) if cost > 0 else 0
                
                positions.append(Position(
                    symbol=pos.get("tradingsymbol", ""),
                    quantity=qty,
                    average_price=avg_price,
                    ltp=ltp,
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    product=pos.get("producttype", "INTRADAY"),
                ))
            
            return positions
        except Exception:
            return []

    def get_balance(self) -> Balance:
        """Get account balance information.
        
        Returns:
            Balance with available funds.
        """
        self._ensure_authenticated()
        
        try:
            response = self._smart_api.rmsLimit()
            
            if not response or not response.get("data"):
                return Balance(
                    available_cash=0.0,
                    used_margin=0.0,
                    total_value=0.0,
                )
            
            data = response["data"]
            available = float(data.get("availablecash", 0))
            used = float(data.get("utiliseddebits", 0))
            
            return Balance(
                available_cash=available,
                used_margin=used,
                total_value=available + used,
            )
        except Exception:
            return Balance(
                available_cash=0.0,
                used_margin=0.0,
                total_value=0.0,
            )
