# Requirements Document

## Introduction

DayTrader is an AI-powered command-line interface for Indian stock market day trading. It combines multi-agent AI systems with Angel One broker integration to provide market research, technical analysis, and trade execution capabilities. The system uses OpenAI Agents SDK for intelligent analysis and SQLite for local data persistence.

## Glossary

- **DayTrader**: The CLI application for day trading
- **Angel One**: Indian stock broker providing SmartAPI for programmatic trading
- **SmartAPI**: Angel One's REST API for trading operations
- **OHLCV**: Open, High, Low, Close, Volume - standard price data format
- **MIS**: Margin Intraday Square-off - intraday trading product type
- **CNC**: Cash and Carry - delivery trading product type
- **LTP**: Last Traded Price
- **SL**: Stop Loss order
- **GTT**: Good Till Triggered order
- **Orchestrator**: Agent that routes queries to appropriate specialized agents
- **Paper Trading**: Simulated trading without real money

## Requirements

### Requirement 1: Configuration Management

**User Story:** As a trader, I want to configure API keys and settings in a separate config file, so that I can securely manage credentials and customize behavior.

#### Acceptance Criteria

1. WHEN the application starts THEN the DayTrader SHALL load configuration from `~/.config/daytrader/config.toml`
2. WHEN configuration file does not exist THEN the DayTrader SHALL create a template config file with placeholder values
3. WHEN API keys are missing or invalid THEN the DayTrader SHALL display a clear error message indicating which keys are required
4. THE DayTrader SHALL support configuration for: OpenAI API key, Angel One API key, Angel One client ID, Angel One PIN, Angel One TOTP secret, default model (gpt-4o), and trading mode (paper/live)
5. WHEN trading mode is set to paper THEN the DayTrader SHALL simulate all trades without connecting to Angel One

### Requirement 2: Broker Authentication

**User Story:** As a trader, I want to authenticate with Angel One automatically, so that I can start trading without manual login each session.

#### Acceptance Criteria

1. WHEN the user runs `daytrader login` THEN the DayTrader SHALL authenticate with Angel One using stored credentials and TOTP
2. WHEN authentication succeeds THEN the DayTrader SHALL store the session token locally for reuse
3. WHEN session token expires THEN the DayTrader SHALL automatically re-authenticate before executing trades
4. WHEN authentication fails THEN the DayTrader SHALL display the specific error from Angel One API
5. WHEN the user runs `daytrader logout` THEN the DayTrader SHALL invalidate the session and clear stored tokens

### Requirement 3: Market Data Retrieval

**User Story:** As a trader, I want to fetch historical and live market data, so that I can analyze price movements and make informed decisions.

#### Acceptance Criteria

1. WHEN the user runs `daytrader data SYMBOL --days N` THEN the DayTrader SHALL fetch N days of OHLCV data from Angel One
2. WHEN data is fetched THEN the DayTrader SHALL store it in the local SQLite database for caching
3. WHEN cached data exists and is recent THEN the DayTrader SHALL use cached data instead of making API calls
4. THE DayTrader SHALL support multiple timeframes: 1min, 5min, 15min, 1hour, 1day
5. WHEN the user runs `daytrader quote SYMBOL` THEN the DayTrader SHALL display current LTP, change, and volume

### Requirement 4: Technical Analysis

**User Story:** As a trader, I want to calculate technical indicators on price data, so that I can identify trading opportunities.

#### Acceptance Criteria

1. WHEN the user runs `daytrader analyze SYMBOL` THEN the DayTrader SHALL calculate and display RSI, MACD, and Bollinger Bands
2. THE DayTrader SHALL support indicators: RSI, MACD, EMA, SMA, Bollinger Bands, ATR, VWAP
3. WHEN calculating indicators THEN the DayTrader SHALL use data from the local SQLite database
4. WHEN the user specifies `--indicators rsi,macd` THEN the DayTrader SHALL calculate only the specified indicators
5. THE DayTrader SHALL display indicator values with buy/sell signal interpretation

### Requirement 5: Stock Screening

**User Story:** As a trader, I want to scan stocks based on technical criteria, so that I can find trading opportunities quickly.

#### Acceptance Criteria

1. WHEN the user runs `daytrader scan --rsi-below 30` THEN the DayTrader SHALL return stocks with RSI below 30
2. WHEN the user runs `daytrader scan --gap-up 2` THEN the DayTrader SHALL return stocks gapping up more than 2%
3. WHEN the user runs `daytrader scan --volume-spike` THEN the DayTrader SHALL return stocks with volume above 2x average
4. THE DayTrader SHALL scan against a configurable watchlist stored in SQLite
5. WHEN scan completes THEN the DayTrader SHALL display results sorted by relevance with key metrics

### Requirement 6: Order Execution

**User Story:** As a trader, I want to place buy and sell orders through the CLI, so that I can execute trades quickly.

#### Acceptance Criteria

1. WHEN the user runs `daytrader buy SYMBOL QTY` THEN the DayTrader SHALL place a market buy order via Angel One API
2. WHEN the user runs `daytrader buy SYMBOL QTY --price PRICE` THEN the DayTrader SHALL place a limit buy order
3. WHEN the user runs `daytrader buy SYMBOL QTY --sl PRICE` THEN the DayTrader SHALL place a stop-loss order after the buy executes
4. WHEN the user runs `daytrader sell SYMBOL QTY` THEN the DayTrader SHALL place a market sell order
5. WHEN order is placed THEN the DayTrader SHALL display order ID, status, and filled price
6. WHEN in paper trading mode THEN the DayTrader SHALL simulate order execution and update local positions
7. THE DayTrader SHALL default to MIS (intraday) product type unless `--delivery` flag is specified

### Requirement 7: Position Management

**User Story:** As a trader, I want to view and manage my open positions, so that I can track my trades and exit when needed.

#### Acceptance Criteria

1. WHEN the user runs `daytrader positions` THEN the DayTrader SHALL display all open positions with current P&L
2. WHEN the user runs `daytrader exit SYMBOL` THEN the DayTrader SHALL close the entire position for that symbol
3. WHEN the user runs `daytrader exit --all` THEN the DayTrader SHALL close all open positions
4. THE DayTrader SHALL display positions with: symbol, quantity, average price, LTP, P&L, P&L percentage
5. WHEN positions are fetched THEN the DayTrader SHALL store them in SQLite for historical tracking

### Requirement 8: Portfolio and P&L Tracking

**User Story:** As a trader, I want to track my portfolio performance and daily P&L, so that I can measure my trading results.

#### Acceptance Criteria

1. WHEN the user runs `daytrader pnl` THEN the DayTrader SHALL display today's realized and unrealized P&L
2. WHEN the user runs `daytrader pnl --history` THEN the DayTrader SHALL display P&L history from SQLite
3. WHEN the user runs `daytrader balance` THEN the DayTrader SHALL display available margin and funds
4. THE DayTrader SHALL automatically log all trades to SQLite with timestamp, symbol, side, quantity, price, and P&L
5. WHEN the user runs `daytrader journal` THEN the DayTrader SHALL display trade history with notes

### Requirement 9: Research Agent

**User Story:** As a trader, I want an AI agent to research stocks using web search, so that I can get comprehensive information quickly.

#### Acceptance Criteria

1. WHEN the user runs `daytrader research SYMBOL` THEN the Research Agent SHALL search the web for recent news and analysis
2. THE Research Agent SHALL use web search tools (Tavily/Serper) to find relevant information
3. WHEN research completes THEN the DayTrader SHALL cache results in SQLite to avoid redundant searches
4. THE Research Agent SHALL summarize findings including: recent news, analyst opinions, key events, and sentiment
5. WHEN the user runs `daytrader research SYMBOL --deep` THEN the Research Agent SHALL perform comprehensive analysis including financials

### Requirement 10: Data Analyst Agent

**User Story:** As a trader, I want an AI agent to analyze price data and identify patterns, so that I can get intelligent insights.

#### Acceptance Criteria

1. WHEN the user runs `daytrader ask "Is RELIANCE oversold?"` THEN the Data Analyst Agent SHALL query SQLite and analyze indicators
2. THE Data Analyst Agent SHALL have tools to: query price data, calculate indicators, identify support/resistance levels
3. WHEN analyzing data THEN the Data Analyst Agent SHALL provide reasoning with specific numbers and levels
4. THE Data Analyst Agent SHALL identify chart patterns: double top/bottom, head and shoulders, triangles
5. WHEN the user asks about entry/exit THEN the Data Analyst Agent SHALL suggest specific price levels with rationale

### Requirement 11: News Agent

**User Story:** As a trader, I want an AI agent to monitor and analyze news sentiment, so that I can react to market-moving events.

#### Acceptance Criteria

1. WHEN the user runs `daytrader news SYMBOL` THEN the News Agent SHALL fetch and summarize recent news
2. THE News Agent SHALL analyze sentiment as: bullish, bearish, or neutral with confidence score
3. WHEN corporate announcements exist THEN the News Agent SHALL highlight earnings, dividends, and corporate actions
4. THE News Agent SHALL cache news in SQLite with timestamp and sentiment score
5. WHEN the user runs `daytrader events --today` THEN the News Agent SHALL list market events affecting watchlist stocks

### Requirement 12: Orchestrator Agent

**User Story:** As a trader, I want to ask natural language questions and have them routed to the right agent, so that I can interact conversationally.

#### Acceptance Criteria

1. WHEN the user runs `daytrader ask "question"` THEN the Orchestrator SHALL determine which agent(s) to invoke
2. THE Orchestrator SHALL route research questions to Research Agent, data questions to Data Analyst, news questions to News Agent
3. WHEN a question requires multiple agents THEN the Orchestrator SHALL combine their responses coherently
4. THE Orchestrator SHALL maintain conversation context within a session
5. WHEN the user asks about trading actions THEN the Orchestrator SHALL confirm before executing any orders

### Requirement 13: Watchlist Management

**User Story:** As a trader, I want to manage a watchlist of stocks, so that I can focus on specific instruments.

#### Acceptance Criteria

1. WHEN the user runs `daytrader watch add SYMBOL` THEN the DayTrader SHALL add the symbol to the watchlist in SQLite
2. WHEN the user runs `daytrader watch remove SYMBOL` THEN the DayTrader SHALL remove the symbol from the watchlist
3. WHEN the user runs `daytrader watch` THEN the DayTrader SHALL display all watchlist symbols with current prices
4. THE DayTrader SHALL support multiple named watchlists: `daytrader watch --list nifty50`
5. WHEN the user runs `daytrader watch import nifty50` THEN the DayTrader SHALL import predefined index constituents

### Requirement 14: Alerts

**User Story:** As a trader, I want to set price alerts, so that I can be notified when stocks reach target levels.

#### Acceptance Criteria

1. WHEN the user runs `daytrader alert SYMBOL "price > 1500"` THEN the DayTrader SHALL store the alert condition in SQLite
2. WHEN the user runs `daytrader alerts` THEN the DayTrader SHALL display all active alerts
3. WHEN an alert condition is met THEN the DayTrader SHALL display a notification in the terminal
4. THE DayTrader SHALL support alert conditions: price above/below, RSI above/below, volume spike
5. WHEN the user runs `daytrader alert remove ID` THEN the DayTrader SHALL delete the specified alert

### Requirement 15: SQLite Data Persistence

**User Story:** As a trader, I want all data stored locally in SQLite, so that I can access historical data offline and reduce API calls.

#### Acceptance Criteria

1. THE DayTrader SHALL store all data in `~/.config/daytrader/daytrader.db`
2. THE DayTrader SHALL create database schema automatically on first run
3. THE DayTrader SHALL store: candles (OHLCV), trades, positions, watchlist, alerts, research cache, journal entries
4. WHEN querying historical data THEN the DayTrader SHALL use SQLite before making API calls
5. THE DayTrader SHALL provide `daytrader db stats` command to show database size and record counts

### Requirement 16: Morning Prep Workflow

**User Story:** As a trader, I want a morning preparation command, so that I can quickly assess market conditions before trading.

#### Acceptance Criteria

1. WHEN the user runs `daytrader prep` THEN the DayTrader SHALL run a comprehensive morning analysis
2. THE prep command SHALL include: gap analysis for watchlist, overnight news summary, FII/DII data if available, key levels for major indices
3. WHEN prep completes THEN the DayTrader SHALL suggest stocks to watch with reasoning
4. THE prep command SHALL use the Research Agent and Data Analyst Agent in combination
5. WHEN market is closed THEN the prep command SHALL indicate market status and next open time

### Requirement 17: End-of-Day Review

**User Story:** As a trader, I want an AI-powered review of my trading day, so that I can learn from my trades.

#### Acceptance Criteria

1. WHEN the user runs `daytrader review` THEN the DayTrader SHALL analyze all trades from today
2. THE review SHALL include: total P&L, win rate, average win/loss, best and worst trades
3. THE Data Analyst Agent SHALL provide insights on what worked and what could improve
4. WHEN the user runs `daytrader review --add-notes "text"` THEN the DayTrader SHALL save notes to the journal
5. THE review SHALL be stored in SQLite for historical reference

### Requirement 18: Paper Trading Mode

**User Story:** As a trader, I want to practice with paper trading, so that I can test strategies without risking real money.

#### Acceptance Criteria

1. WHEN trading mode is set to paper in config THEN the DayTrader SHALL simulate all order executions
2. THE paper trading mode SHALL maintain virtual balance and positions in SQLite
3. WHEN paper order is placed THEN the DayTrader SHALL execute at current market price with simulated slippage
4. THE DayTrader SHALL track paper trading P&L separately from live trading
5. WHEN the user runs `daytrader paper reset` THEN the DayTrader SHALL reset paper trading balance to configured starting amount
