"""Base utilities for AI agents.

This module provides common utilities for creating and running AI agents
using the OpenAI Agents SDK.
"""

import os
from typing import Any, Callable, Optional

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
