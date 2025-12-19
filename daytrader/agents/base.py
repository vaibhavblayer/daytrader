"""Base utilities for AI agents.

This module provides common utilities for creating and running AI agents
using the OpenAI Agents SDK.
"""

import os
from typing import Any, Callable, Optional

# Disable tracing to avoid noisy 503 errors from telemetry
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

from agents import Agent, Runner


# Default model to use for agents
DEFAULT_MODEL = "gpt-5.2"


def get_model() -> str:
    """Get the model to use for agents.
    
    Checks OPENAI_MODEL environment variable, falls back to default.
    
    Returns:
        Model name string.
    """
    return os.environ.get("OPENAI_MODEL", DEFAULT_MODEL)


def get_api_key() -> Optional[str]:
    """Get the OpenAI API key.
    
    Returns:
        API key string or None if not configured.
    """
    return os.environ.get("OPENAI_API_KEY")


def create_agent(
    name: str,
    instructions: str,
    tools: Optional[list[Callable[..., Any]]] = None,
    model: Optional[str] = None,
) -> Agent:
    """Create an AI agent with the specified configuration.
    
    Args:
        name: Name of the agent.
        instructions: System instructions for the agent.
        tools: Optional list of tool functions the agent can use.
        model: Optional model override. Uses default if not specified.
        
    Returns:
        Configured Agent instance.
    """
    agent_model = model or get_model()
    
    return Agent(
        name=name,
        instructions=instructions,
        tools=tools or [],
        model=agent_model,
    )


def _get_model_info(model: str) -> tuple[str, str]:
    """Get model display name and reasoning level.
    
    Args:
        model: Model name string.
        
    Returns:
        Tuple of (display_name, reasoning_level).
    """
    # O-series reasoning models
    o_series = {
        "o1": ("o1", "high"),
        "o1-mini": ("o1-mini", "medium"),
        "o1-preview": ("o1-preview", "high"),
        "o3": ("o3", "very high"),
        "o3-mini": ("o3-mini", "medium"),
        "o4-mini": ("o4-mini", "medium"),
    }
    
    if model in o_series:
        return o_series[model]
    
    # GPT-5 series with reasoning support
    if model.startswith("gpt-5"):
        # gpt-5.2, gpt-5.1, gpt-5, etc.
        return (model, "medium")
    
    # GPT-4 series
    if model.startswith("gpt-4"):
        return (model, "standard")
    
    # GPT-3.5 and older
    if model.startswith("gpt-"):
        return (model, "basic")
    
    return (model, "unknown")


def _log_agent_call(agent: Agent) -> None:
    """Log agent call info to terminal.
    
    Args:
        agent: The agent being called.
    """
    from rich.console import Console
    
    console = Console()
    model = agent.model
    display_name, reasoning = _get_model_info(model)
    
    console.print(
        f"[dim]🤖 Agent: {agent.name} | Model: {display_name} | Reasoning: {reasoning}[/dim]"
    )


def run_agent_sync(
    agent: Agent,
    message: str,
    context: Optional[dict[str, Any]] = None,
) -> str:
    """Run an agent synchronously and return the response.
    
    Args:
        agent: The agent to run.
        message: User message to send to the agent.
        context: Optional context dictionary to pass to the agent.
        
    Returns:
        Agent's response as a string.
    """
    _log_agent_call(agent)
    result = Runner.run_sync(agent, message, context=context)
    return result.final_output


async def run_agent_async(
    agent: Agent,
    message: str,
    context: Optional[dict[str, Any]] = None,
) -> str:
    """Run an agent asynchronously and return the response.
    
    Args:
        agent: The agent to run.
        message: User message to send to the agent.
        context: Optional context dictionary to pass to the agent.
        
    Returns:
        Agent's response as a string.
    """
    _log_agent_call(agent)
    result = await Runner.run(agent, message, context=context)
    return result.final_output


def function_tool(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to mark a function as an agent tool.
    
    This is a pass-through decorator that can be used to mark functions
    as tools for documentation purposes. The OpenAI Agents SDK automatically
    converts functions to tools.
    
    Args:
        func: Function to mark as a tool.
        
    Returns:
        The same function, unchanged.
    """
    return func
