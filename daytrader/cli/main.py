"""Main CLI entry point for DayTrader.

This module provides the main click group and lazy loading
for heavy imports to improve startup time.
"""

import click
from rich.console import Console

# Console for rich output
console = Console()


class LazyGroup(click.Group):
    """A click Group that lazily loads commands.
    
    This improves CLI startup time by only importing
    command modules when they are actually invoked.
    """

    def __init__(self, *args, lazy_subcommands: dict[str, str] | None = None, **kwargs):
        """Initialize the lazy group.
        
        Args:
            lazy_subcommands: Mapping of command names to module paths.
        """
        super().__init__(*args, **kwargs)
        self._lazy_subcommands = lazy_subcommands or {}

    def list_commands(self, ctx: click.Context) -> list[str]:
        """List all available commands."""
        base = super().list_commands(ctx)
        lazy = list(self._lazy_subcommands.keys())
        return sorted(set(base + lazy))

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        """Get a command by name, lazily loading if needed."""
        # First check if it's already loaded
        if cmd_name in self.commands:
            return self.commands[cmd_name]
        
        # Check if it's a lazy command
        if cmd_name in self._lazy_subcommands:
            return self._lazy_load(cmd_name)
        
        return None

    def _lazy_load(self, cmd_name: str) -> click.Command:
        """Lazily load a command from its module path."""
        import importlib
        
        module_path = self._lazy_subcommands[cmd_name]
        module = importlib.import_module(module_path)
        
        # The command should be named the same as cmd_name or be the default
        if hasattr(module, cmd_name):
            attr = getattr(module, cmd_name)
            if isinstance(attr, click.Command):
                cmd = attr
            else:
                raise click.ClickException(f"'{cmd_name}' in {module_path} is not a click command")
        else:
            # Look for a command with the same name
            cmd = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, click.Command) and attr.name == cmd_name:
                    cmd = attr
                    break
            
            if cmd is None:
                raise click.ClickException(f"Could not find command '{cmd_name}' in {module_path}")
        
        self.add_command(cmd)
        return cmd


# Define lazy subcommands mapping
LAZY_SUBCOMMANDS = {
    "login": "daytrader.cli.auth",
    "logout": "daytrader.cli.auth",
    "data": "daytrader.cli.data",
    "quote": "daytrader.cli.data",
    "live": "daytrader.cli.data",
    "analyze": "daytrader.cli.analyze",
    "signal": "daytrader.cli.analyze",
    "mtf": "daytrader.cli.analyze",
    "scan": "daytrader.cli.scan",
    "buy": "daytrader.cli.trade",
    "sell": "daytrader.cli.trade",
    "positions": "daytrader.cli.trade",
    "exit": "daytrader.cli.trade",
    "pnl": "daytrader.cli.portfolio",
    "balance": "daytrader.cli.portfolio",
    "journal": "daytrader.cli.portfolio",
    "sync": "daytrader.cli.portfolio",
    "report": "daytrader.cli.portfolio",
    # AI Features
    "ask": "daytrader.cli.ask",
    "research": "daytrader.cli.ask",
    "news": "daytrader.cli.ask",
    "events": "daytrader.cli.ask",
    # Watchlist and Alerts
    "watch": "daytrader.cli.watchlist",
    "alert": "daytrader.cli.alerts",
    "alerts": "daytrader.cli.alerts",
    # Workflows
    "prep": "daytrader.cli.workflow",
    "review": "daytrader.cli.workflow",
    # Paper Trading
    "paper": "daytrader.cli.paper",
    # Discovery
    "discover": "daytrader.cli.discover",
}


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(cls=LazyGroup, lazy_subcommands=LAZY_SUBCOMMANDS, context_settings=CONTEXT_SETTINGS)
@click.version_option(package_name="daytrader")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """DayTrader - AI-powered CLI for Indian stock market day trading.
    
    Use this CLI to research stocks, analyze technical indicators,
    execute trades, and manage your portfolio through natural language
    interaction with AI agents.
    
    \b
    Quick Start:
      daytrader login          # Authenticate with Angel One
      daytrader ask "question" # Ask AI about stocks
      daytrader positions      # View open positions
    """
    # Ensure context object exists for passing data between commands
    ctx.ensure_object(dict)


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
