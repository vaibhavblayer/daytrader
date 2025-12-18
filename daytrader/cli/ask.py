"""Ask command for DayTrader CLI.

Routes natural language queries to the Orchestrator Agent
for intelligent responses using multiple specialized agents.
"""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


def _get_config():
    """Lazily load configuration."""
    import toml
    
    config_path = Path.home() / ".config" / "daytrader" / "config.toml"
    
    if not config_path.exists():
        return None
    
    try:
        return toml.load(config_path)
    except Exception:
        return None


def _get_db_path() -> Path:
    """Get the database path."""
    return Path.home() / ".config" / "daytrader" / "daytrader.db"


@click.command()
@click.argument("query")
def ask(query: str) -> None:
    """Ask the AI assistant a question about stocks or trading.
    
    QUERY is your natural language question. The orchestrator will
    route it to the appropriate specialized agent(s).
    
    \b
    Agent routing:
      - Research questions → Research Agent
      - Technical analysis → Data Analyst Agent
      - News and sentiment → News Agent
      - Trading actions   → Trading Agent
    
    \b
    Examples:
      daytrader ask "Is RELIANCE oversold?"
      daytrader ask "What's the latest news on TCS?"
      daytrader ask "Research INFY for me"
      daytrader ask "What are the support levels for HDFCBANK?"
    """
    config = _get_config()
    
    if config is None:
        console.print(Panel(
            "[red]Configuration not found.[/red]\n\n"
            "Run [cyan]daytrader login[/cyan] to create a config file.",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    # Check for OpenAI API key
    openai_key = config.get("openai", {}).get("api_key", "")
    if not openai_key or openai_key == "your-openai-api-key":
        console.print(Panel(
            "[red]OpenAI API key not configured.[/red]\n\n"
            "Please add your OpenAI API key to:\n"
            f"[cyan]{Path.home() / '.config' / 'daytrader' / 'config.toml'}[/cyan]",
            title="[bold red]Configuration Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    console.print(f"[dim]Processing your query...[/dim]\n")
    
    try:
        from daytrader.agents.orchestrator import OrchestratorAgent
        
        db_path = _get_db_path()
        orchestrator = OrchestratorAgent(db_path)
        
        # Show which agents will be used
        agents = orchestrator.get_agent_for_query(query)
        agent_names = [a.title() for a in agents]
        console.print(f"[dim]Routing to: {', '.join(agent_names)} Agent(s)[/dim]\n")
        
        # Get response
        response = orchestrator.ask(query)
        
        # Display response as markdown
        console.print(Panel(
            Markdown(response),
            title="[bold cyan]AI Response[/bold cyan]",
            border_style="cyan",
        ))
        
    except ImportError as e:
        console.print(Panel(
            f"[red]Missing dependency: {e}[/red]\n\n"
            "Please install the required packages:\n"
            "[cyan]pip install openai-agents[/cyan]",
            title="[bold red]Import Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    except Exception as e:
        console.print(Panel(
            f"[red]Error processing query: {e}[/red]",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)


@click.command()
@click.argument("symbol")
@click.option(
    "--deep",
    is_flag=True,
    help="Perform comprehensive research including financials.",
)
def research(symbol: str, deep: bool) -> None:
    """Research a stock using AI-powered web search.
    
    SYMBOL is the trading symbol (e.g., RELIANCE, INFY, TCS).
    
    The Research Agent will search for:
    - Recent news and developments
    - Analyst opinions and price targets
    - Key events and catalysts
    
    Use --deep for comprehensive analysis including financials.
    
    \b
    Examples:
      daytrader research RELIANCE          # Basic research
      daytrader research TCS --deep        # Deep research with financials
    """
    config = _get_config()
    
    if config is None:
        console.print(Panel(
            "[red]Configuration not found.[/red]\n\n"
            "Run [cyan]daytrader login[/cyan] to create a config file.",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    # Check for OpenAI API key
    openai_key = config.get("openai", {}).get("api_key", "")
    if not openai_key or openai_key == "your-openai-api-key":
        console.print(Panel(
            "[red]OpenAI API key not configured.[/red]\n\n"
            "Please add your OpenAI API key to:\n"
            f"[cyan]{Path.home() / '.config' / 'daytrader' / 'config.toml'}[/cyan]",
            title="[bold red]Configuration Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    symbol = symbol.upper()
    research_type = "comprehensive" if deep else "basic"
    
    console.print(f"[dim]Researching {symbol} ({research_type} analysis)...[/dim]\n")
    
    try:
        from daytrader.agents.research import ResearchAgent
        
        db_path = _get_db_path()
        agent = ResearchAgent(db_path)
        
        # Perform research
        response = agent.research(symbol, deep=deep)
        
        # Display response as markdown
        title = f"[bold cyan]Research: {symbol}[/bold cyan]"
        if deep:
            title += " [dim](Deep Analysis)[/dim]"
        
        console.print(Panel(
            Markdown(response),
            title=title,
            border_style="cyan",
        ))
        
    except ImportError as e:
        console.print(Panel(
            f"[red]Missing dependency: {e}[/red]\n\n"
            "Please install the required packages:\n"
            "[cyan]pip install openai-agents[/cyan]",
            title="[bold red]Import Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    except Exception as e:
        console.print(Panel(
            f"[red]Error during research: {e}[/red]",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)


@click.command()
@click.argument("symbol")
def news(symbol: str) -> None:
    """Get and analyze news for a stock.
    
    SYMBOL is the trading symbol (e.g., RELIANCE, INFY, TCS).
    
    The News Agent will:
    - Search for recent news about the stock
    - Analyze sentiment (bullish/bearish/neutral)
    - Highlight key events and announcements
    
    \b
    Examples:
      daytrader news RELIANCE
      daytrader news TCS
    """
    config = _get_config()
    
    if config is None:
        console.print(Panel(
            "[red]Configuration not found.[/red]\n\n"
            "Run [cyan]daytrader login[/cyan] to create a config file.",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    # Check for OpenAI API key
    openai_key = config.get("openai", {}).get("api_key", "")
    if not openai_key or openai_key == "your-openai-api-key":
        console.print(Panel(
            "[red]OpenAI API key not configured.[/red]\n\n"
            "Please add your OpenAI API key to:\n"
            f"[cyan]{Path.home() / '.config' / 'daytrader' / 'config.toml'}[/cyan]",
            title="[bold red]Configuration Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    symbol = symbol.upper()
    
    console.print(f"[dim]Fetching news for {symbol}...[/dim]\n")
    
    try:
        from daytrader.agents.news import NewsAgent
        
        db_path = _get_db_path()
        agent = NewsAgent(db_path)
        
        # Get news
        response = agent.get_news(symbol)
        
        # Display response as markdown
        console.print(Panel(
            Markdown(response),
            title=f"[bold cyan]News: {symbol}[/bold cyan]",
            border_style="cyan",
        ))
        
    except ImportError as e:
        console.print(Panel(
            f"[red]Missing dependency: {e}[/red]\n\n"
            "Please install the required packages:\n"
            "[cyan]pip install openai-agents[/cyan]",
            title="[bold red]Import Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    except Exception as e:
        console.print(Panel(
            f"[red]Error fetching news: {e}[/red]",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)


@click.command()
@click.option(
    "--today",
    is_flag=True,
    help="Show events for today only.",
)
def events(today: bool) -> None:
    """Get market events and announcements.
    
    Shows market events, economic data releases, and corporate
    announcements that may affect stocks.
    
    \b
    Examples:
      daytrader events          # General market events
      daytrader events --today  # Today's events only
    """
    config = _get_config()
    
    if config is None:
        console.print(Panel(
            "[red]Configuration not found.[/red]\n\n"
            "Run [cyan]daytrader login[/cyan] to create a config file.",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    # Check for OpenAI API key
    openai_key = config.get("openai", {}).get("api_key", "")
    if not openai_key or openai_key == "your-openai-api-key":
        console.print(Panel(
            "[red]OpenAI API key not configured.[/red]\n\n"
            "Please add your OpenAI API key to:\n"
            f"[cyan]{Path.home() / '.config' / 'daytrader' / 'config.toml'}[/cyan]",
            title="[bold red]Configuration Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    from datetime import date as date_type
    
    event_date = date_type.today() if today else None
    date_str = event_date.strftime("%B %d, %Y") if event_date else "upcoming"
    
    console.print(f"[dim]Fetching {date_str} market events...[/dim]\n")
    
    try:
        from daytrader.agents.news import NewsAgent
        
        db_path = _get_db_path()
        agent = NewsAgent(db_path)
        
        # Get events
        response = agent.get_events(event_date)
        
        # Display response as markdown
        title = "[bold cyan]Market Events[/bold cyan]"
        if today:
            title += f" [dim]({date_str})[/dim]"
        
        console.print(Panel(
            Markdown(response),
            title=title,
            border_style="cyan",
        ))
        
    except ImportError as e:
        console.print(Panel(
            f"[red]Missing dependency: {e}[/red]\n\n"
            "Please install the required packages:\n"
            "[cyan]pip install openai-agents[/cyan]",
            title="[bold red]Import Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    except Exception as e:
        console.print(Panel(
            f"[red]Error fetching events: {e}[/red]",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
