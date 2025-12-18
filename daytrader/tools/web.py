"""Web search tools for AI agents.

These tools allow agents to search the web for stock research,
news, and other information using the Tavily API.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from daytrader.db.store import DataStore


# Default database path for caching
DEFAULT_DB_PATH = Path.home() / ".config" / "daytrader" / "daytrader.db"


def _get_data_store(db_path: Optional[Path] = None) -> DataStore:
    """Get a DataStore instance."""
    return DataStore(db_path or DEFAULT_DB_PATH)


def _get_tavily_client():
    """Get a Tavily client instance.
    
    Checks environment variable first, then config file.
    
    Returns:
        TavilyClient instance or None if API key not configured.
    """
    # Check env var first
    api_key = os.environ.get("TAVILY_API_KEY")
    
    # Fall back to config file
    if not api_key:
        try:
            import toml
            config_path = Path.home() / ".config" / "daytrader" / "config.toml"
            if config_path.exists():
                config = toml.load(config_path)
                api_key = config.get("tavily", {}).get("api_key")
        except Exception:
            pass
    
    if not api_key:
        return None
    
    try:
        from tavily import TavilyClient
        return TavilyClient(api_key=api_key)
    except ImportError:
        return None


def web_search(
    query: str,
    search_depth: str = "basic",
    max_results: int = 5,
    include_domains: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
    cache_results: bool = True,
    cache_hours: int = 24,
    db_path: Optional[Path] = None,
) -> dict:
    """Search the web using Tavily API.
    
    This tool performs web searches to find relevant information about
    stocks, companies, market news, and financial analysis.
    
    Args:
        query: Search query string.
        search_depth: Search depth - "basic" or "advanced" (more thorough).
        max_results: Maximum number of results to return (default 5).
        include_domains: List of domains to include in search.
        exclude_domains: List of domains to exclude from search.
        cache_results: Whether to cache results in database (default True).
        cache_hours: Hours to cache results (default 24).
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - query: The search query
        - results: List of search result dictionaries with title, url, content
        - count: Number of results
        - cached: Whether results were from cache
        - error: Error message if search failed (None if successful)
    """
    try:
        store = _get_data_store(db_path)
        
        # Check cache first
        if cache_results:
            cached_content = store.get_cached_research(
                symbol=f"__search__{query}",
                max_age_hours=cache_hours,
            )
            if cached_content:
                import json
                try:
                    cached_results = json.loads(cached_content)
                    return {
                        "query": query,
                        "results": cached_results,
                        "count": len(cached_results),
                        "cached": True,
                        "error": None,
                    }
                except json.JSONDecodeError:
                    pass  # Cache corrupted, proceed with fresh search
        
        # Get Tavily client
        client = _get_tavily_client()
        if client is None:
            return {
                "query": query,
                "results": [],
                "count": 0,
                "cached": False,
                "error": "Tavily API key not configured. Set TAVILY_API_KEY environment variable.",
            }
        
        # Perform search
        search_params = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
        }
        
        if include_domains:
            search_params["include_domains"] = include_domains
        if exclude_domains:
            search_params["exclude_domains"] = exclude_domains
        
        response = client.search(**search_params)
        
        # Extract results
        results = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0),
            })
        
        # Cache results
        if cache_results and results:
            import json
            store.cache_research(
                symbol=f"__search__{query}",
                content=json.dumps(results),
                source="tavily",
            )
        
        return {
            "query": query,
            "results": results,
            "count": len(results),
            "cached": False,
            "error": None,
        }
        
    except Exception as e:
        return {
            "query": query,
            "results": [],
            "count": 0,
            "cached": False,
            "error": str(e),
        }


def search_stock_news(
    symbol: str,
    company_name: Optional[str] = None,
    days: int = 7,
    max_results: int = 10,
    cache_hours: int = 6,
    db_path: Optional[Path] = None,
) -> dict:
    """Search for recent news about a stock.
    
    This tool searches for recent news articles about a specific stock
    or company.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE", "TCS").
        company_name: Optional company name for better search results.
        days: Number of days of news to search (default 7).
        max_results: Maximum number of results (default 10).
        cache_hours: Hours to cache results (default 6).
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - symbol: The stock symbol
        - news: List of news articles
        - count: Number of articles
        - cached: Whether results were from cache
        - error: Error message if search failed (None if successful)
    """
    # Build search query
    if company_name:
        query = f"{company_name} ({symbol}) stock news India"
    else:
        query = f"{symbol} stock news India NSE BSE"
    
    # Use financial news domains
    include_domains = [
        "moneycontrol.com",
        "economictimes.indiatimes.com",
        "livemint.com",
        "business-standard.com",
        "ndtvprofit.com",
        "reuters.com",
        "bloomberg.com",
    ]
    
    result = web_search(
        query=query,
        search_depth="basic",
        max_results=max_results,
        include_domains=include_domains,
        cache_results=True,
        cache_hours=cache_hours,
        db_path=db_path,
    )
    
    return {
        "symbol": symbol.upper(),
        "news": result.get("results", []),
        "count": result.get("count", 0),
        "cached": result.get("cached", False),
        "error": result.get("error"),
    }


def search_company_info(
    symbol: str,
    company_name: Optional[str] = None,
    include_financials: bool = False,
    cache_hours: int = 24,
    db_path: Optional[Path] = None,
) -> dict:
    """Search for company information and analysis.
    
    This tool searches for company information, analyst reports,
    and financial analysis.
    
    Args:
        symbol: Stock symbol (e.g., "RELIANCE", "TCS").
        company_name: Optional company name for better search results.
        include_financials: Whether to include financial data search.
        cache_hours: Hours to cache results (default 24).
        db_path: Optional path to database file.
        
    Returns:
        Dictionary containing:
        - symbol: The stock symbol
        - company_info: List of company information results
        - financials: List of financial data results (if requested)
        - cached: Whether results were from cache
        - error: Error message if search failed (None if successful)
    """
    results = {
        "symbol": symbol.upper(),
        "company_info": [],
        "financials": [],
        "cached": False,
        "error": None,
    }
    
    # Search for company info
    if company_name:
        query = f"{company_name} ({symbol}) company profile analysis India"
    else:
        query = f"{symbol} company profile analysis NSE India"
    
    info_result = web_search(
        query=query,
        search_depth="advanced",
        max_results=5,
        cache_results=True,
        cache_hours=cache_hours,
        db_path=db_path,
    )
    
    results["company_info"] = info_result.get("results", [])
    results["cached"] = info_result.get("cached", False)
    
    if info_result.get("error"):
        results["error"] = info_result["error"]
        return results
    
    # Search for financials if requested
    if include_financials:
        if company_name:
            fin_query = f"{company_name} ({symbol}) quarterly results earnings financials"
        else:
            fin_query = f"{symbol} quarterly results earnings financials India"
        
        fin_result = web_search(
            query=fin_query,
            search_depth="basic",
            max_results=5,
            cache_results=True,
            cache_hours=cache_hours,
            db_path=db_path,
        )
        
        results["financials"] = fin_result.get("results", [])
        
        if fin_result.get("error"):
            results["error"] = fin_result["error"]
    
    return results
