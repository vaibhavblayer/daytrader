# DayTrader

AI-powered CLI for Indian stock market day trading. Research stocks, analyze technical indicators, execute trades, and manage your portfolio through natural language interaction with AI agents.

## Features

- **AI-Powered Analysis**: Ask questions about stocks using natural language
- **Technical Indicators**: Built-in technical analysis with pandas-ta
- **Angel One Integration**: Trade directly through Angel One broker
- **Paper Trading**: Practice strategies without risking real money
- **Portfolio Management**: Track positions, P&L, and journal trades
- **Watchlists & Alerts**: Monitor stocks and set price alerts
- **Market Scanning**: Discover trading opportunities
- **Workflow Automation**: Pre-market prep and post-market review

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

Set up your environment variables:

```bash
export OPENAI_API_KEY="your-openai-key"
export TAVILY_API_KEY="your-tavily-key"  # For news/research
```

## Usage

### Authentication

```bash
# Login to Angel One
daytrader login

# Logout
daytrader logout
```

### AI Assistant

```bash
# Ask questions about stocks
daytrader ask "What's the trend for RELIANCE?"

# Research a stock
daytrader research INFY

# Get latest news
daytrader news TATAMOTORS

# Check upcoming events
daytrader events HDFCBANK
```

### Market Data

```bash
# Get stock quote
daytrader quote RELIANCE

# Fetch historical data
daytrader data INFY --days 30
```

### Technical Analysis

```bash
# Analyze a stock
daytrader analyze RELIANCE

# Scan for opportunities
daytrader scan --pattern bullish
```

### Trading

```bash
# Buy shares
daytrader buy RELIANCE --qty 10

# Sell shares
daytrader sell RELIANCE --qty 10

# View positions
daytrader positions

# Exit all positions
daytrader exit
```

### Paper Trading

```bash
# Practice trading without real money
daytrader paper buy RELIANCE --qty 10
daytrader paper positions
```

### Portfolio

```bash
# Check P&L
daytrader pnl

# View account balance
daytrader balance

# Trade journal
daytrader journal
```

### Watchlist & Alerts

```bash
# Add to watchlist
daytrader watch add RELIANCE

# Set price alert
daytrader alert RELIANCE --above 2500

# View alerts
daytrader alerts
```

### Workflows

```bash
# Pre-market preparation
daytrader prep

# Post-market review
daytrader review
```

### Discovery

```bash
# Discover trading opportunities
daytrader discover
```

## Development

```bash
# Install dev dependencies
poetry install --with dev

# Run tests
pytest

# Run specific test file
pytest tests/test_agents.py
```

## Project Structure

```
daytrader/
├── agents/       # AI agents (analyst, trader, research, news)
├── brokers/      # Broker integrations (Angel One, paper trading)
├── cli/          # CLI commands
├── db/           # Data storage
├── indicators/   # Technical indicators
├── models/       # Data models
└── tools/        # Agent tools
```

## License

MIT
