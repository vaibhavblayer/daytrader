# Design Document

## Overview

DayTrader is a Python CLI application for Indian stock market day trading. It uses a multi-agent AI architecture built on OpenAI Agents SDK, with Angel One SmartAPI for broker integration and SQLite for local data persistence. The system provides market research, technical analysis, and trade execution through natural language interaction.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              CLI Layer                                   │
│  (click commands: login, data, analyze, scan, buy, sell, ask, etc.)     │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                          Orchestrator Agent                              │
│              (Routes queries to appropriate agents)                      │
└───────┬─────────────┬─────────────┬─────────────┬───────────────────────┘
        │             │             │             │
        ▼             ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│ Research  │  │   Data    │  │   News    │  │  Trading  │
│   Agent   │  │  Analyst  │  │   Agent   │  │   Agent   │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │              │              │              │
      ▼              ▼              ▼              ▼
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│Web Search │  │  SQLite   │  │ News APIs │  │  Broker   │
│  (Tavily) │  │    DB     │  │           │  │   API     │
└───────────┘  └───────────┘  └───────────┘  └───────────┘
```

## Components and Interfaces

### 1. Configuration Module (`config.py`)

```python
@dataclass
class Config:
    # OpenAI
    openai_api_key: str
    model: str = "gpt-4o"
    
    # Angel One
    angelone_api_key: str
    angelone_client_id: str
    angelone_pin: str
    angelone_totp_secret: str
    
    # Web Search
    tavily_api_key: str
    
    # Trading
    trading_mode: Literal["paper", "live"] = "paper"
    paper_starting_balance: float = 100000.0
    default_product: Literal["MIS", "CNC"] = "MIS"
    
    # Paths
    config_dir: Path
    db_path: Path

def load_config() -> Config: ...
def save_config(config: Config) -> None: ...
def create_template_config() -> None: ...
```

### 2. Database Module (`db/store.py`)

```python
class DataStore:
    def __init__(self, db_path: Path): ...
    
    # Candles
    def save_candles(self, symbol: str, timeframe: str, candles: list[Candle]) -> None: ...
    def get_candles(self, symbol: str, timeframe: str, from_date: date, to_date: date) -> list[Candle]: ...
    
    # Trades
    def log_trade(self, trade: Trade) -> None: ...
    def get_trades(self, date: Optional[date] = None) -> list[Trade]: ...
    
    # Positions
    def save_position(self, position: Position) -> None: ...
    def get_positions() -> list[Position]: ...
    
    # Watchlist
    def add_to_watchlist(self, symbol: str, list_name: str = "default") -> None: ...
    def remove_from_watchlist(self, symbol: str, list_name: str = "default") -> None: ...
    def get_watchlist(self, list_name: str = "default") -> list[str]: ...
    
    # Alerts
    def save_alert(self, alert: Alert) -> int: ...
    def get_alerts() -> list[Alert]: ...
    def delete_alert(self, alert_id: int) -> None: ...
    
    # Research Cache
    def cache_research(self, symbol: str, content: str, source: str) -> None: ...
    def get_cached_research(self, symbol: str, max_age_hours: int = 24) -> Optional[str]: ...
    
    # Journal
    def save_journal_entry(self, entry: JournalEntry) -> None: ...
    def get_journal(self, from_date: Optional[date] = None) -> list[JournalEntry]: ...
```

### 3. Broker Module (`brokers/`)

```python
# brokers/base.py
class BaseBroker(ABC):
    @abstractmethod
    def login(self) -> bool: ...
    
    @abstractmethod
    def get_quote(self, symbol: str) -> Quote: ...
    
    @abstractmethod
    def get_historical(self, symbol: str, from_date: date, to_date: date, interval: str) -> list[Candle]: ...
    
    @abstractmethod
    def place_order(self, order: Order) -> OrderResult: ...
    
    @abstractmethod
    def get_positions(self) -> list[Position]: ...
    
    @abstractmethod
    def get_balance(self) -> Balance: ...
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

# brokers/angelone.py
class AngelOneBroker(BaseBroker):
    def __init__(self, config: Config): ...
    # Implements all abstract methods using SmartAPI

# brokers/paper.py
class PaperBroker(BaseBroker):
    def __init__(self, config: Config, data_store: DataStore): ...
    # Simulates trading with virtual balance
```

### 4. Indicators Module (`indicators/technical.py`)

```python
def calculate_rsi(prices: list[float], period: int = 14) -> list[float]: ...
def calculate_macd(prices: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[list, list, list]: ...
def calculate_ema(prices: list[float], period: int) -> list[float]: ...
def calculate_sma(prices: list[float], period: int) -> list[float]: ...
def calculate_bollinger_bands(prices: list[float], period: int = 20, std_dev: float = 2.0) -> tuple[list, list, list]: ...
def calculate_atr(high: list[float], low: list[float], close: list[float], period: int = 14) -> list[float]: ...
def calculate_vwap(high: list[float], low: list[float], close: list[float], volume: list[float]) -> list[float]: ...
```

### 5. Agents Module (`agents/`)

```python
# agents/base.py
def create_agent(name: str, instructions: str, tools: list) -> Agent: ...
def run_agent_sync(agent: Agent, message: str) -> str: ...

# agents/orchestrator.py
class OrchestratorAgent:
    """Routes queries to appropriate specialized agents."""
    
    def route_query(self, query: str) -> str: ...
    def get_agent_for_query(self, query: str) -> str: ...  # Returns agent name

# agents/research.py
class ResearchAgent:
    """Performs web research on stocks."""
    
    tools = [web_search, get_company_info, get_financials]
    
    def research(self, symbol: str, deep: bool = False) -> str: ...

# agents/analyst.py
class DataAnalystAgent:
    """Analyzes price data and identifies patterns."""
    
    tools = [query_candles, calculate_indicators, find_support_resistance, identify_patterns]
    
    def analyze(self, symbol: str, query: str) -> str: ...

# agents/news.py
class NewsAgent:
    """Fetches and analyzes news sentiment."""
    
    tools = [search_news, get_announcements, analyze_sentiment]
    
    def get_news(self, symbol: str) -> str: ...
    def get_events(self, date: date) -> str: ...

# agents/trader.py
class TradingAgent:
    """Executes trades with confirmation."""
    
    tools = [place_order, get_positions, get_balance, cancel_order]
    
    def execute_trade(self, action: str, symbol: str, qty: int, **kwargs) -> str: ...
```

### 6. CLI Module (`cli/`)

```python
# cli/main.py
@click.group()
def cli(): ...

# cli/auth.py
@cli.command()
def login(): ...

@cli.command()
def logout(): ...

# cli/data.py
@cli.command()
@click.argument("symbol")
@click.option("--days", default=30)
@click.option("--timeframe", default="1day")
def data(symbol, days, timeframe): ...

@cli.command()
@click.argument("symbol")
def quote(symbol): ...

# cli/analyze.py
@cli.command()
@click.argument("symbol")
@click.option("--indicators", default="rsi,macd,bb")
def analyze(symbol, indicators): ...

# cli/scan.py
@cli.command()
@click.option("--rsi-below", type=float)
@click.option("--rsi-above", type=float)
@click.option("--gap-up", type=float)
@click.option("--gap-down", type=float)
@click.option("--volume-spike", is_flag=True)
def scan(**filters): ...

# cli/trade.py
@cli.command()
@click.argument("symbol")
@click.argument("qty", type=int)
@click.option("--price", type=float)
@click.option("--sl", type=float)
@click.option("--target", type=float)
@click.option("--delivery", is_flag=True)
def buy(symbol, qty, price, sl, target, delivery): ...

@cli.command()
@click.argument("symbol")
@click.argument("qty", type=int)
@click.option("--price", type=float)
def sell(symbol, qty, price): ...

@cli.command()
def positions(): ...

@cli.command()
@click.argument("symbol", required=False)
@click.option("--all", "exit_all", is_flag=True)
def exit(symbol, exit_all): ...

# cli/portfolio.py
@cli.command()
@click.option("--history", is_flag=True)
def pnl(history): ...

@cli.command()
def balance(): ...

@cli.command()
def journal(): ...

# cli/ask.py
@cli.command()
@click.argument("query")
def ask(query): ...

@cli.command()
@click.argument("symbol")
@click.option("--deep", is_flag=True)
def research(symbol, deep): ...

@cli.command()
@click.argument("symbol")
def news(symbol): ...

# cli/watchlist.py
@cli.group()
def watch(): ...

@watch.command()
@click.argument("symbol")
def add(symbol): ...

@watch.command()
@click.argument("symbol")
def remove(symbol): ...

@watch.command()
@click.argument("name")
def import_list(name): ...

# cli/alerts.py
@cli.command()
@click.argument("symbol")
@click.argument("condition")
def alert(symbol, condition): ...

@cli.command()
def alerts(): ...

# cli/workflow.py
@cli.command()
def prep(): ...

@cli.command()
@click.option("--add-notes", type=str)
def review(add_notes): ...

# cli/paper.py
@cli.group()
def paper(): ...

@paper.command()
def reset(): ...

@paper.command()
def status(): ...
```

## Data Models

```python
# models/candle.py
@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

# models/order.py
@dataclass
class Order:
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    order_type: Literal["MARKET", "LIMIT", "SL", "SL-M"]
    product: Literal["MIS", "CNC"]
    price: Optional[float] = None
    trigger_price: Optional[float] = None

@dataclass
class OrderResult:
    order_id: str
    status: str
    filled_qty: int
    filled_price: float
    message: str

# models/position.py
@dataclass
class Position:
    symbol: str
    quantity: int
    average_price: float
    ltp: float
    pnl: float
    pnl_percent: float
    product: str

# models/trade.py
@dataclass
class Trade:
    id: Optional[int]
    timestamp: datetime
    symbol: str
    side: str
    quantity: int
    price: float
    order_id: str
    pnl: Optional[float]
    is_paper: bool

# models/alert.py
@dataclass
class Alert:
    id: Optional[int]
    symbol: str
    condition: str  # e.g., "price > 1500", "rsi < 30"
    created_at: datetime
    triggered: bool = False

# models/journal.py
@dataclass
class JournalEntry:
    date: date
    trades_count: int
    total_pnl: float
    win_rate: float
    notes: Optional[str]
    ai_insights: Optional[str]
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Configuration Loading Consistency
*For any* valid config file, loading and then saving the config should produce an equivalent file (round-trip consistency).
**Validates: Requirements 1.1**

### Property 2: Missing Config Key Detection
*For any* config file with missing required keys, the system should identify and report all missing keys.
**Validates: Requirements 1.3**

### Property 3: Paper Mode Isolation
*For any* trade executed in paper mode, no external API calls to Angel One should be made.
**Validates: Requirements 1.5**

### Property 4: Session Token Persistence
*For any* successful authentication, the session token should be retrievable from storage until explicitly invalidated.
**Validates: Requirements 2.2**

### Property 5: Token Expiry Re-authentication
*For any* expired session token, the system should automatically re-authenticate before executing the requested operation.
**Validates: Requirements 2.3**

### Property 6: Data Fetch Completeness
*For any* data request for N days, the returned candles should cover the requested date range (excluding market holidays).
**Validates: Requirements 3.1**

### Property 7: Cache Persistence
*For any* fetched data, querying the same data immediately after should return results from cache without API calls.
**Validates: Requirements 3.2, 3.3**

### Property 8: Timeframe Support
*For any* supported timeframe (1min, 5min, 15min, 1hour, 1day), data fetching should succeed and return correctly aggregated candles.
**Validates: Requirements 3.4**

### Property 9: RSI Calculation Accuracy
*For any* price series, the calculated RSI should be within acceptable tolerance of reference implementation (pandas-ta).
**Validates: Requirements 4.1, 4.2**

### Property 10: MACD Calculation Accuracy
*For any* price series, the calculated MACD, signal, and histogram should match reference implementation.
**Validates: Requirements 4.1, 4.2**

### Property 11: Indicator Data Source
*For any* indicator calculation, the data should be sourced from SQLite database.
**Validates: Requirements 4.3**

### Property 12: Selective Indicator Calculation
*For any* indicator filter specification, only the specified indicators should be calculated.
**Validates: Requirements 4.4**

### Property 13: RSI Filter Accuracy
*For any* RSI threshold filter, all returned stocks should have RSI values satisfying the condition.
**Validates: Requirements 5.1**

### Property 14: Gap Filter Accuracy
*For any* gap percentage filter, all returned stocks should have gap values satisfying the condition.
**Validates: Requirements 5.2**

### Property 15: Volume Spike Detection
*For any* volume spike filter, all returned stocks should have volume above 2x their average.
**Validates: Requirements 5.3**

### Property 16: Watchlist Scan Scope
*For any* scan operation, only stocks in the configured watchlist should be scanned.
**Validates: Requirements 5.4**

### Property 17: Order Execution in Paper Mode
*For any* order placed in paper mode, the virtual balance and positions should be updated correctly.
**Validates: Requirements 6.6**

### Property 18: Default Product Type
*For any* order without explicit product flag, the product type should default to MIS.
**Validates: Requirements 6.7**

### Property 19: Position Retrieval Completeness
*For any* open positions query, all positions should be returned with current P&L calculated.
**Validates: Requirements 7.1**

### Property 20: Position Exit Completeness
*For any* exit command for a symbol, the entire position should be closed.
**Validates: Requirements 7.2**

### Property 21: Exit All Completeness
*For any* exit --all command, all open positions should be closed.
**Validates: Requirements 7.3**

### Property 22: Trade Logging Completeness
*For any* executed trade, a corresponding entry should exist in the trades table.
**Validates: Requirements 8.4**

### Property 23: P&L Calculation Accuracy
*For any* set of trades, the calculated P&L should equal sum of (exit_price - entry_price) * quantity for each trade.
**Validates: Requirements 8.1**

### Property 24: Research Cache Persistence
*For any* research query, the results should be cached and retrievable within the cache TTL.
**Validates: Requirements 9.3**

### Property 25: News Sentiment Classification
*For any* news analysis, the sentiment should be classified as one of: bullish, bearish, neutral.
**Validates: Requirements 11.2**

### Property 26: News Cache Persistence
*For any* news query, the results should be cached in SQLite.
**Validates: Requirements 11.4**

### Property 27: Orchestrator Routing Consistency
*For any* research-related query, the Orchestrator should route to the Research Agent.
**Validates: Requirements 12.2**

### Property 28: Multi-Agent Coordination
*For any* query requiring multiple agents, all relevant agents should be invoked and responses combined.
**Validates: Requirements 12.3**

### Property 29: Conversation Context Persistence
*For any* follow-up query in a session, previous context should be available to the agent.
**Validates: Requirements 12.4**

### Property 30: Trade Confirmation Requirement
*For any* query that would execute a trade, confirmation should be required before execution.
**Validates: Requirements 12.5**

### Property 31: Watchlist Add/Remove Consistency
*For any* symbol added to watchlist, it should be retrievable; after removal, it should not be retrievable.
**Validates: Requirements 13.1, 13.2, 13.3**

### Property 32: Multiple Watchlist Support
*For any* named watchlist, symbols should be stored and retrieved independently from other lists.
**Validates: Requirements 13.4**

### Property 33: Alert Storage and Retrieval
*For any* created alert, it should be retrievable until deleted.
**Validates: Requirements 14.1, 14.2**

### Property 34: Alert Condition Evaluation
*For any* alert with price condition, the alert should trigger when the condition is met.
**Validates: Requirements 14.3**

### Property 35: Alert Deletion
*For any* deleted alert, it should no longer be retrievable.
**Validates: Requirements 14.5**

### Property 36: Database Schema Completeness
*For any* fresh database, all required tables (candles, trades, positions, watchlist, alerts, research_cache, journal) should exist.
**Validates: Requirements 15.3**

### Property 37: Cache-First Data Access
*For any* data query with cached data available, the cache should be used before making API calls.
**Validates: Requirements 15.4**

### Property 38: Prep Agent Invocation
*For any* prep command, both Research Agent and Data Analyst Agent should be invoked.
**Validates: Requirements 16.4**

### Property 39: Market Status Check
*For any* prep command when market is closed, the market status should be indicated.
**Validates: Requirements 16.5**

### Property 40: Review Metrics Calculation
*For any* review command, total P&L, win rate, and average win/loss should be calculated from today's trades.
**Validates: Requirements 17.1, 17.2**

### Property 41: Journal Note Persistence
*For any* review with notes, the notes should be saved and retrievable.
**Validates: Requirements 17.4**

### Property 42: Paper Trading Balance Tracking
*For any* paper trade, the virtual balance should be updated correctly (decreased on buy, increased on sell).
**Validates: Requirements 18.2**

### Property 43: Paper Trading Slippage Simulation
*For any* paper market order, the execution price should include simulated slippage.
**Validates: Requirements 18.3**

### Property 44: Paper Trading Separation
*For any* paper trade, it should be tracked separately from live trades.
**Validates: Requirements 18.4**

### Property 45: Paper Trading Reset
*For any* paper reset command, the balance should be reset to the configured starting amount and all paper positions cleared.
**Validates: Requirements 18.5**

## Error Handling

1. **Network Errors**: Retry with exponential backoff for transient failures
2. **Authentication Errors**: Clear session and prompt for re-login
3. **Rate Limiting**: Respect Angel One rate limits (10 req/sec), queue excess requests
4. **Invalid Symbols**: Validate symbols against exchange master before operations
5. **Insufficient Funds**: Check balance before placing orders, display clear error
6. **Market Hours**: Check market status before placing orders, warn if market closed

## Testing Strategy

### Unit Tests
- Config loading/saving
- Indicator calculations against known values
- Order model validation
- Alert condition parsing
- P&L calculations

### Property-Based Tests (Hypothesis)
- Config round-trip consistency
- Indicator calculation accuracy
- Filter logic correctness
- Watchlist operations
- Alert condition evaluation
- Paper trading balance updates

### Integration Tests
- Database operations
- Agent tool invocations
- CLI command execution
- Broker API mocking

### Testing Framework
- **pytest** for test runner
- **hypothesis** for property-based testing
- **pytest-mock** for mocking
- **click.testing.CliRunner** for CLI tests
