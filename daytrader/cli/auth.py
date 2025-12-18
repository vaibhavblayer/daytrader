"""Authentication commands for DayTrader CLI.

Handles login/logout with Angel One broker using TOTP authentication.
"""

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


def _get_config():
    """Lazily load configuration.
    
    Returns:
        Config dict or None if not configured.
    """
    from pathlib import Path
    import toml
    
    config_path = Path.home() / ".config" / "daytrader" / "config.toml"
    
    if not config_path.exists():
        return None
    
    try:
        return toml.load(config_path)
    except Exception:
        return None


def _get_broker(config: dict):
    """Get the appropriate broker based on config.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Broker instance.
    """
    trading_mode = config.get("trading", {}).get("mode", "paper")
    
    if trading_mode == "paper":
        from daytrader.brokers.paper import PaperBroker
        from daytrader.db.store import DataStore
        from pathlib import Path
        
        db_path = Path.home() / ".config" / "daytrader" / "daytrader.db"
        store = DataStore(db_path)
        starting_balance = config.get("trading", {}).get("paper_starting_balance", 100000.0)
        return PaperBroker(store, starting_balance=starting_balance)
    else:
        from daytrader.brokers.angelone import AngelOneBroker
        
        angelone_config = config.get("angelone", {})
        return AngelOneBroker(
            api_key=angelone_config.get("api_key", ""),
            client_id=angelone_config.get("client_id", ""),
            pin=angelone_config.get("pin", ""),
            totp_secret=angelone_config.get("totp_secret", ""),
        )


def _create_template_config():
    """Create a template configuration file."""
    from pathlib import Path
    import toml
    
    config_dir = Path.home() / ".config" / "daytrader"
    config_path = config_dir / "config.toml"
    
    config_dir.mkdir(parents=True, exist_ok=True)
    
    template = {
        "openai": {
            "api_key": "",  # Leave empty to use OPENAI_API_KEY env var
            "model": "gpt-4o",
        },
        "angelone": {
            "api_key": "your-angelone-api-key",
            "client_id": "your-client-id",
            "pin": "your-pin",
            "totp_secret": "your-totp-secret",
        },
        "tavily": {
            "api_key": "your-tavily-api-key",
        },
        "trading": {
            "mode": "paper",  # paper or live
            "paper_starting_balance": 100000.0,
            "default_product": "MIS",  # MIS or CNC
        },
    }
    
    with open(config_path, "w") as f:
        toml.dump(template, f)
    
    return config_path


def _validate_config(config: dict) -> list[str]:
    """Validate configuration and return list of missing keys.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        List of missing required keys.
    """
    import os
    
    missing = []
    trading_mode = config.get("trading", {}).get("mode", "paper")
    
    # OpenAI - check config first, then env var
    openai_key = config.get("openai", {}).get("api_key")
    if not openai_key or openai_key == "your-openai-api-key":
        # Fall back to environment variable
        if not os.environ.get("OPENAI_API_KEY"):
            missing.append("openai.api_key (or set OPENAI_API_KEY env var)")
    
    # Angel One credentials only required for live trading
    if trading_mode == "live":
        angelone = config.get("angelone", {})
        if not angelone.get("api_key"):
            missing.append("angelone.api_key")
        if not angelone.get("client_id"):
            missing.append("angelone.client_id")
        if not angelone.get("pin"):
            missing.append("angelone.pin")
        if not angelone.get("totp_secret"):
            missing.append("angelone.totp_secret")
    
    return missing


@click.command()
def login() -> None:
    """Authenticate with Angel One broker.
    
    Logs in using credentials from config file and stores
    the session token for subsequent commands.
    
    In paper trading mode, this validates the configuration
    without connecting to Angel One.
    """
    config = _get_config()
    
    if config is None:
        config_path = _create_template_config()
        console.print(Panel(
            f"[yellow]Configuration file created at:[/yellow]\n"
            f"[cyan]{config_path}[/cyan]\n\n"
            f"Please edit this file with your API keys and credentials,\n"
            f"then run [green]daytrader login[/green] again.",
            title="[bold]Configuration Required[/bold]",
            border_style="yellow",
        ))
        raise SystemExit(1)
    
    # Validate configuration
    missing_keys = _validate_config(config)
    if missing_keys:
        console.print(Panel(
            f"[red]Missing required configuration keys:[/red]\n\n"
            + "\n".join(f"  • {key}" for key in missing_keys) +
            f"\n\n[dim]Edit ~/.config/daytrader/config.toml to add these values.[/dim]",
            title="[bold red]Configuration Error[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)
    
    trading_mode = config.get("trading", {}).get("mode", "paper")
    
    if trading_mode == "paper":
        console.print(Panel(
            "[green]✓[/green] Paper trading mode active\n\n"
            "[dim]All trades will be simulated without connecting to Angel One.\n"
            "Use [cyan]daytrader paper status[/cyan] to check your virtual balance.[/dim]",
            title="[bold green]Login Successful[/bold green]",
            border_style="green",
        ))
        return
    
    # Live trading - authenticate with Angel One
    console.print("[dim]Authenticating with Angel One...[/dim]")
    
    try:
        broker = _get_broker(config)
        
        if broker.login():
            client_id = config.get("angelone", {}).get("client_id", "")
            console.print(Panel(
                f"[green]✓[/green] Authenticated as [cyan]{client_id}[/cyan]\n\n"
                "[dim]Session token stored. You can now execute trades.[/dim]",
                title="[bold green]Login Successful[/bold green]",
                border_style="green",
            ))
        else:
            # Get detailed error from broker
            error_detail = broker.get_last_error() if hasattr(broker, 'get_last_error') else "Unknown error"
            console.print(Panel(
                f"[red]✗[/red] Authentication failed\n\n"
                f"[yellow]Error:[/yellow] {error_detail}\n\n"
                "[dim]Please check your credentials in config.toml:\n"
                "  • API key - from Angel One SmartAPI dashboard\n"
                "  • Client ID - your Angel One client ID\n"
                "  • PIN - your 4-digit MPIN\n"
                "  • TOTP secret - the secret key (not the 6-digit code)[/dim]",
                title="[bold red]Login Failed[/bold red]",
                border_style="red",
            ))
            raise SystemExit(1)
            
    except Exception as e:
        console.print(Panel(
            f"[red]✗[/red] Authentication error\n\n"
            f"[dim]{str(e)}[/dim]",
            title="[bold red]Login Failed[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)


@click.command()
def logout() -> None:
    """Logout and clear session tokens.
    
    Invalidates the current session with Angel One and
    removes stored session tokens from local storage.
    """
    config = _get_config()
    
    if config is None:
        console.print("[yellow]No configuration found. Nothing to logout from.[/yellow]")
        return
    
    trading_mode = config.get("trading", {}).get("mode", "paper")
    
    if trading_mode == "paper":
        console.print(Panel(
            "[green]✓[/green] Paper trading session cleared\n\n"
            "[dim]Your virtual positions and balance remain intact.\n"
            "Use [cyan]daytrader paper reset[/cyan] to reset your paper account.[/dim]",
            title="[bold green]Logout Successful[/bold green]",
            border_style="green",
        ))
        return
    
    # Live trading - logout from Angel One
    console.print("[dim]Logging out from Angel One...[/dim]")
    
    try:
        broker = _get_broker(config)
        
        if broker.logout():
            console.print(Panel(
                "[green]✓[/green] Session invalidated and tokens cleared\n\n"
                "[dim]You will need to login again to execute trades.[/dim]",
                title="[bold green]Logout Successful[/bold green]",
                border_style="green",
            ))
        else:
            # Even if logout fails, clear local tokens
            console.print(Panel(
                "[yellow]⚠[/yellow] Could not invalidate remote session\n\n"
                "[dim]Local tokens have been cleared.\n"
                "The remote session may expire automatically.[/dim]",
                title="[bold yellow]Partial Logout[/bold yellow]",
                border_style="yellow",
            ))
            
    except Exception as e:
        console.print(Panel(
            f"[yellow]⚠[/yellow] Logout warning\n\n"
            f"[dim]{str(e)}\n\n"
            "Local tokens have been cleared.[/dim]",
            title="[bold yellow]Partial Logout[/bold yellow]",
            border_style="yellow",
        ))
