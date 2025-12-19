# DayTrader

AI-powered CLI for Indian stock market day trading. Research stocks, analyze technical indicators, execute trades, and manage your portfolio.

## Features

- **Technical Analysis**: RSI, MACD, Bollinger Bands, SuperTrend, Stochastic, ADX, Fibonacci, Pivot Points, and more
- **Signal Scoring**: Composite buy/sell signals (-100 to +100)
- **Multi-Timeframe Analysis**: Analyze across 5min, 15min, 1hour, 1day
- **Angel One Integration**: Trade directly through Angel One broker
- **Paper Trading**: Practice strategies without risking real money
- **Live Price Watching**: Real-time price monitoring
- **Trade Sync & Reporting**: Import trades from Angel One, analyze performance
- **Market Hours Awareness**: Warns about MIS auto square-off at 3:15 PM

## Installation

Requires Python 3.12+

```bash
# Clone the repository
git clone https://github.com/vaibhavblayer/daytrader.git
cd daytrader

# Install with poetry
poetry install

# Or install with pip
pip install .
```

## Configuration

Create config file at `~/.config/daytrader/config.toml`:

```toml
[trading]
mode = "live"  # "paper" for paper trading
paper_starting_balance = 100000.0

[angelone]
api_key = "your-angel-one-api-key"
client_id = "your-client-id"
pin = "your-pin"
totp_secret = "your-totp-secret"

[openai]
api_key = "your-openai-key"  # Optional, for AI features

[tavily]
api_key = "your-tavily-key"  # Optional, for news/research
```

### Getting Angel One API Credentials

1. Login to [Angel One SmartAPI](https://smartapi.angelone.in/)
2. Create an app to get your API key
3. Your client ID is your Angel One login ID
4. PIN is your trading PIN
5. TOTP secret is from your authenticator app setup

## Usage

### Authentication

```bash
# Login to Angel One (generates session)
daytrader login

# Logout
daytrader logout
```

### Market Data

```bash
# Get stock quote
daytrader quote RELIANCE

# Fetch historical data (default 30 days)
daytrader data INFY --days 60

# Watch live prices (refreshes every 2 seconds)
daytrader live YESBANK IDEA RELIANCE --refresh 2
```

### Technical Analysis

```bash
# Default analysis (Bollinger Bands, SuperTrend, Fibonacci, RSI, ADX)
daytrader analyze RELIANCE

# With specific indicators (comma-separated)
daytrader analyze RELIANCE --indicators rsi,macd
daytrader analyze RELIANCE -i supertrend,adx,fib
daytrader analyze RELIANCE -i pivot,fib,patterns

# Available indicators:
#   rsi, macd, ema, sma, bb (Bollinger), atr, vwap,
#   stoch (Stochastic), adx, supertrend, fib (Fibonacci),
#   pivot, obv, cci, willr (Williams %R), patterns

# Signal score (-100 to +100, combines multiple indicators)
daytrader signal YESBANK

# Multi-timeframe analysis (5min, 15min, 1hour, 1day)
daytrader mtf RELIANCE
```

### Trading

```bash
# Buy shares (market order, intraday/MIS)
daytrader buy RELIANCE 10

# Buy with limit price
daytrader buy RELIANCE 10 --price 2500

# Buy with stop-loss
daytrader buy RELIANCE 10 --sl 2450

# Buy for delivery (CNC)
daytrader buy RELIANCE 10 --delivery

# After Market Order (AMO)
daytrader buy RELIANCE 10 --amo

# Sell shares
daytrader sell RELIANCE 10
daytrader sell RELIANCE 10 --price 2600
daytrader sell RELIANCE 10 --delivery

# View open positions (shows quick sell commands)
daytrader positions

# Exit specific position
daytrader exit RELIANCE

# Exit all positions
daytrader exit --all
```

### Portfolio & Reporting

```bash
# Check account balance
daytrader balance

# View trade journal
daytrader journal

# Sync trades from Angel One
daytrader sync

# Performance report (win rate, P&L analysis)
daytrader report
```

### Market Scanning

```bash
# Scan for opportunities
daytrader scan --pattern bullish
```

### AI Assistant (requires OpenAI key)

```bash
# Ask questions about stocks
daytrader ask "What's the trend for RELIANCE?"

# Research a stock
daytrader research INFY
```

## Market Hours

The CLI is aware of Indian market hours:
- **Market Open**: 9:15 AM - 3:00 PM
- **MIS Warning**: 3:00 PM - 3:15 PM (square-off window)
- **MIS Closed**: After 3:15 PM (auto square-off happens)
- **Market Closed**: After 3:30 PM, weekends, holidays

## Development

```bash
# Install dev dependencies
poetry install --with dev

# Run tests
pytest

# Run specific test
pytest tests/test_indicators.py -v
```

## Project Structure

```
daytrader/
├── agents/       # AI agents
├── brokers/      # Angel One, paper trading
├── cli/          # CLI commands
├── db/           # SQLite data storage
├── indicators/   # Technical indicators
├── models/       # Data models
└── tools/        # Agent tools
```

## License

MIT
