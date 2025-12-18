"""Workflow commands for DayTrader CLI.

Handles morning prep and end-of-day review workflows.
"""

from datetime import date, datetime, time
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()


# Agent types used in prep workflow
PREP_REQUIRED_AGENTS = ["research", "analyst"]


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


def _get_data_store():
    """Get the data store instance."""
    from daytrader.db.store import DataStore
    
    return DataStore(_get_db_path())


def is_market_open() -> tuple[bool, str]:
    """Check if Indian stock market is open.
    
    Returns:
        Tuple of (is_open, status_message).
    """
    now = datetime.now()
    current_time = now.time()
    
    # Market hours: 9:15 AM to 3:30 PM IST
    market_open = time(9, 15)
    market_close = time(15, 30)
    
    # Check if weekend
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        next_open = "Monday 9:15 AM"
        return False, f"Market closed (Weekend). Next open: {next_open}"
    
    # Check market hours
    if current_time < market_open:
        return False, f"Market opens at 9:15 AM (Pre-market)"
    elif current_time > market_close:
        return False, "Market closed for today (Post-market)"
    else:
        return True, "Market is OPEN"


def get_market_status() -> dict:
    """Get detailed market status.
    
    Returns:
        Dictionary with market status information.
    """
    is_open, message = is_market_open()
    now = datetime.now()
    
    return {
        "is_open": is_open,
        "message": message,
        "current_time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "day": now.strftime("%A"),
    }


def get_prep_agents() -> list[str]:
    """Get the list of agents that will be invoked during prep.
    
    This function returns the agent types that the prep command
    will invoke for morning analysis.
    
    Returns:
        List of agent type names.
    """
    return PREP_REQUIRED_AGENTS.copy()


def run_prep_analysis(
    watchlist: list[str],
    db_path: Path,
) -> dict:
    """Run prep analysis using Research and Data Analyst agents.
    
    This is the core prep logic extracted for testability.
    
    Args:
        watchlist: List of stock symbols to analyze.
        db_path: Path to the database.
        
    Returns:
        Dictionary with analysis results and agents invoked.
    """
    from daytrader.agents.research import ResearchAgent
    from daytrader.agents.analyst import DataAnalystAgent
    
    agents_invoked = []
    results = {}
    
    # Run Research Agent
    research_agent = ResearchAgent(db_path)
    agents_invoked.append("research")
    
    symbols_str = ", ".join(watchlist[:5])
    research_query = f"""Provide a brief morning market summary for Indian markets.
    
Focus on:
1. Key overnight global market movements (US, Asia)
2. Any major news affecting these stocks: {symbols_str}
3. Important economic events or announcements today

Keep it concise and actionable for day trading."""
    
    results["research"] = research_agent.ask(research_query)
    
    # Run Data Analyst Agent
    analyst_agent = DataAnalystAgent(db_path)
    agents_invoked.append("analyst")
    
    analyst_query = f"""Provide a quick technical overview for day trading.

For these stocks: {symbols_str}

Include:
1. Key support and resistance levels
2. Any gap up/down from previous close
3. RSI status (oversold/overbought)
4. Stocks to watch today with brief reasoning

Keep analysis brief and actionable."""
    
    results["analyst"] = analyst_agent.ask(analyst_query)
    
    return {
        "agents_invoked": agents_invoked,
        "results": results,
        "watchlist_analyzed": watchlist[:5],
    }


def calculate_review_metrics(trades: list) -> dict:
    """Calculate review metrics from a list of trades.
    
    Args:
        trades: List of Trade objects.
        
    Returns:
        Dictionary with review metrics.
    """
    if not trades:
        return {
            "total_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "best_trade": None,
            "worst_trade": None,
            "total_volume": 0,
        }
    
    total_pnl = 0.0
    winning_trades = 0
    losing_trades = 0
    total_wins = 0.0
    total_losses = 0.0
    best_trade = None
    worst_trade = None
    best_pnl = float('-inf')
    worst_pnl = float('inf')
    total_volume = 0
    
    for trade in trades:
        total_volume += trade.quantity * trade.price
        
        if trade.pnl is not None:
            total_pnl += trade.pnl
            
            if trade.pnl > 0:
                winning_trades += 1
                total_wins += trade.pnl
                if trade.pnl > best_pnl:
                    best_pnl = trade.pnl
                    best_trade = trade
            elif trade.pnl < 0:
                losing_trades += 1
                total_losses += abs(trade.pnl)
                if trade.pnl < worst_pnl:
                    worst_pnl = trade.pnl
                    worst_trade = trade
    
    total_trades = len(trades)
    trades_with_pnl = winning_trades + losing_trades
    
    win_rate = (winning_trades / trades_with_pnl * 100) if trades_with_pnl > 0 else 0.0
    avg_win = (total_wins / winning_trades) if winning_trades > 0 else 0.0
    avg_loss = (total_losses / losing_trades) if losing_trades > 0 else 0.0
    
    return {
        "total_pnl": total_pnl,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "total_volume": total_volume,
    }


@click.command()
def prep() -> None:
    """Run morning preparation analysis.
    
    Performs comprehensive morning analysis including:
    - Market status check
    - Gap analysis for watchlist stocks
    - Overnight news summary
    - Key levels for major indices
    - Stocks to watch with reasoning
    
    Uses Research Agent and Data Analyst Agent for analysis.
    
    \b
    Examples:
      daytrader prep
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
    
    console.print("[bold cyan]Morning Prep Analysis[/bold cyan]\n")
    
    # 1. Check market status
    market_status = get_market_status()
    status_color = "green" if market_status["is_open"] else "yellow"
    
    console.print(Panel(
        f"[bold]{market_status['day']}, {market_status['date']}[/bold]\n"
        f"Time: {market_status['current_time']}\n"
        f"Status: [{status_color}]{market_status['message']}[/{status_color}]",
        title="[bold]Market Status[/bold]",
        border_style=status_color,
    ))
    
    # 2. Get watchlist
    store = _get_data_store()
    watchlist = store.get_watchlist()
    
    if not watchlist:
        console.print("\n[dim]No stocks in watchlist. Add stocks with:[/dim]")
        console.print("[cyan]daytrader watch add SYMBOL[/cyan]\n")
        watchlist = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"]
        console.print(f"[dim]Using default stocks: {', '.join(watchlist)}[/dim]\n")
    else:
        console.print(f"\n[dim]Analyzing watchlist: {', '.join(watchlist[:10])}{'...' if len(watchlist) > 10 else ''}[/dim]\n")
    
    # 3. Run agents for analysis
    console.print("[dim]Running AI analysis...[/dim]\n")
    
    try:
        # Import agents
        from daytrader.agents.research import ResearchAgent
        from daytrader.agents.analyst import DataAnalystAgent
        
        db_path = _get_db_path()
        
        # Run Research Agent for news summary
        console.print("[dim]→ Research Agent: Gathering overnight news...[/dim]")
        research_agent = ResearchAgent(db_path)
        
        # Build research query
        symbols_str = ", ".join(watchlist[:5])  # Limit to first 5 for speed
        research_query = f"""Provide a brief morning market summary for Indian markets.
        
Focus on:
1. Key overnight global market movements (US, Asia)
2. Any major news affecting these stocks: {symbols_str}
3. Important economic events or announcements today

Keep it concise and actionable for day trading."""
        
        research_result = research_agent.ask(research_query)
        
        console.print(Panel(
            Markdown(research_result),
            title="[bold cyan]Market & News Summary[/bold cyan]",
            border_style="cyan",
        ))
        
        # Run Data Analyst Agent for technical levels
        console.print("\n[dim]→ Data Analyst Agent: Analyzing key levels...[/dim]")
        analyst_agent = DataAnalystAgent(db_path)
        
        analyst_query = f"""Provide a quick technical overview for day trading.

For these stocks: {symbols_str}

Include:
1. Key support and resistance levels
2. Any gap up/down from previous close
3. RSI status (oversold/overbought)
4. Stocks to watch today with brief reasoning

Keep analysis brief and actionable."""
        
        analyst_result = analyst_agent.ask(analyst_query)
        
        console.print(Panel(
            Markdown(analyst_result),
            title="[bold cyan]Technical Analysis[/bold cyan]",
            border_style="cyan",
        ))
        
        # Summary
        console.print(Panel(
            "[bold]Prep Complete![/bold]\n\n"
            "Use these commands to dig deeper:\n"
            "• [cyan]daytrader analyze SYMBOL[/cyan] - Detailed technical analysis\n"
            "• [cyan]daytrader news SYMBOL[/cyan] - Latest news for a stock\n"
            "• [cyan]daytrader scan --rsi-below 30[/cyan] - Find oversold stocks",
            title="[bold green]Ready to Trade[/bold green]",
            border_style="green",
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
            f"[red]Error during prep analysis: {e}[/red]",
            title="[bold red]Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)


@click.command()
@click.option(
    "--add-notes",
    type=str,
    default=None,
    help="Add notes to today's journal entry.",
)
def review(add_notes: Optional[str]) -> None:
    """Review today's trading performance.
    
    Analyzes all trades from today and provides:
    - Total P&L and trade statistics
    - Win rate and average win/loss
    - Best and worst trades
    - AI-powered insights on what worked and what could improve
    
    Use --add-notes to save personal notes to the journal.
    
    \b
    Examples:
      daytrader review                    # Review today's trades
      daytrader review --add-notes "Good discipline today"
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
    
    store = _get_data_store()
    today = date.today()
    
    console.print(f"[bold cyan]End-of-Day Review[/bold cyan] - {today.strftime('%Y-%m-%d')}\n")
    
    # Get today's trades
    trades = store.get_trades(trade_date=today)
    
    if not trades:
        console.print(Panel(
            "[dim]No trades recorded for today.[/dim]\n\n"
            "Trades are logged automatically when you execute orders.",
            title="[bold]No Trades[/bold]",
            border_style="dim",
        ))
        
        # Still save journal entry if notes provided
        if add_notes:
            from daytrader.models.journal import JournalEntry
            
            entry = JournalEntry(
                date=today,
                trades_count=0,
                total_pnl=0.0,
                win_rate=0.0,
                notes=add_notes,
                ai_insights=None,
            )
            store.save_journal_entry(entry)
            console.print(f"\n[green]Notes saved to journal.[/green]")
        
        return
    
    # Calculate metrics
    metrics = calculate_review_metrics(trades)
    
    # Display metrics
    pnl_color = "green" if metrics["total_pnl"] >= 0 else "red"
    pnl_sign = "+" if metrics["total_pnl"] >= 0 else ""
    
    metrics_text = (
        f"[bold]Performance Summary[/bold]\n\n"
        f"Total P&L:      [{pnl_color}]{pnl_sign}₹{metrics['total_pnl']:.2f}[/{pnl_color}]\n"
        f"Total Trades:   {metrics['total_trades']}\n"
        f"Winning Trades: [green]{metrics['winning_trades']}[/green]\n"
        f"Losing Trades:  [red]{metrics['losing_trades']}[/red]\n"
        f"Win Rate:       {metrics['win_rate']:.1f}%\n"
        f"{'─' * 30}\n"
        f"Avg Win:        [green]₹{metrics['avg_win']:.2f}[/green]\n"
        f"Avg Loss:       [red]₹{metrics['avg_loss']:.2f}[/red]\n"
        f"Total Volume:   ₹{metrics['total_volume']:,.2f}"
    )
    
    console.print(Panel(
        metrics_text,
        title="[bold cyan]Trade Statistics[/bold cyan]",
        border_style="cyan",
    ))
    
    # Best and worst trades
    if metrics["best_trade"]:
        best = metrics["best_trade"]
        console.print(f"\n[green]Best Trade:[/green] {best.symbol} {best.side} {best.quantity} @ ₹{best.price:.2f} → +₹{best.pnl:.2f}")
    
    if metrics["worst_trade"]:
        worst = metrics["worst_trade"]
        console.print(f"[red]Worst Trade:[/red] {worst.symbol} {worst.side} {worst.quantity} @ ₹{worst.price:.2f} → ₹{worst.pnl:.2f}")
    
    # Trade list
    console.print("\n")
    table = Table(
        title="Today's Trades",
        show_header=True,
        header_style="bold",
    )
    
    table.add_column("Time", style="dim")
    table.add_column("Symbol", style="bold")
    table.add_column("Side", justify="center")
    table.add_column("Qty", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("P&L", justify="right")
    
    for trade in trades:
        side_color = "green" if trade.side == "BUY" else "red"
        
        if trade.pnl is not None:
            pnl_color = "green" if trade.pnl >= 0 else "red"
            pnl_sign = "+" if trade.pnl >= 0 else ""
            pnl_str = f"[{pnl_color}]{pnl_sign}₹{trade.pnl:.2f}[/{pnl_color}]"
        else:
            pnl_str = "-"
        
        table.add_row(
            trade.timestamp.strftime("%H:%M"),
            trade.symbol,
            f"[{side_color}]{trade.side}[/{side_color}]",
            str(trade.quantity),
            f"₹{trade.price:.2f}",
            pnl_str,
        )
    
    console.print(table)
    
    # AI Insights (optional - only if OpenAI key configured)
    ai_insights = None
    openai_key = config.get("openai", {}).get("api_key", "")
    
    if openai_key and openai_key != "your-openai-api-key":
        console.print("\n[dim]Generating AI insights...[/dim]")
        
        try:
            from daytrader.agents.analyst import DataAnalystAgent
            
            db_path = _get_db_path()
            analyst = DataAnalystAgent(db_path)
            
            # Build trade summary for AI
            trade_summary = []
            for t in trades:
                pnl_str = f"P&L: ₹{t.pnl:.2f}" if t.pnl else "P&L: N/A"
                trade_summary.append(f"- {t.side} {t.quantity} {t.symbol} @ ₹{t.price:.2f} ({pnl_str})")
            
            insight_query = f"""Analyze these trades from today and provide brief insights:

{chr(10).join(trade_summary)}

Total P&L: ₹{metrics['total_pnl']:.2f}
Win Rate: {metrics['win_rate']:.1f}%

Provide:
1. What worked well (1-2 points)
2. What could improve (1-2 points)
3. One actionable tip for tomorrow

Keep it brief and constructive."""
            
            ai_insights = analyst.ask(insight_query)
            
            console.print(Panel(
                Markdown(ai_insights),
                title="[bold cyan]AI Insights[/bold cyan]",
                border_style="cyan",
            ))
            
        except Exception as e:
            console.print(f"[dim]Could not generate AI insights: {e}[/dim]")
    
    # Save to journal
    from daytrader.models.journal import JournalEntry
    
    entry = JournalEntry(
        date=today,
        trades_count=metrics["total_trades"],
        total_pnl=metrics["total_pnl"],
        win_rate=metrics["win_rate"],
        notes=add_notes,
        ai_insights=ai_insights,
    )
    store.save_journal_entry(entry)
    
    console.print(f"\n[green]Review saved to journal.[/green]")
    
    if add_notes:
        console.print(f"[dim]Notes: {add_notes}[/dim]")
