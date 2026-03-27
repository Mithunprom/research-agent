"""
tools_web.py – Web search and page fetching tools for the research agent.

Provides two capabilities:
  1. Semantic web search via Tavily API (returns ranked results with snippets)
  2. Direct URL fetching with HTML→plain-text extraction via BeautifulSoup

Flow:
    Agent needs external information
        → tavily_search(query) hits the Tavily search API
            → returns top-K results with titles, URLs, and snippets
        → Agent picks a promising URL from the results
        → fetch_text(url) downloads the page, strips HTML, returns clean text
        → Agent incorporates the text into its research notes via tools_memory
"""

import os
import requests
from bs4 import BeautifulSoup


def tavily_search(query: str, k: int = 5):
    """Search the web using the Tavily API and return top-K results.

    Flow:
        query string (e.g. "FAISS vector indexing explained")
            → Read TAVILY_API_KEY from environment (raises RuntimeError if missing)
            → POST to https://api.tavily.com/search with query + max_results
            → Parse JSON response
            → Return list of result dicts, each containing:
                  title, url, content (snippet), score

    Args:
        query: natural language search query
        k: maximum number of results to return (default 5)

    Returns:
        list[dict]: search results from Tavily, each with title/url/content/score

    Raises:
        RuntimeError: if TAVILY_API_KEY is not set
        requests.HTTPError: if the API returns a non-2xx status
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("Missing TAVILY_API_KEY in environment/.env")

    url = "https://api.tavily.com/search"
    payload = {"api_key": api_key, "query": query, "max_results": k}
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data.get("results", [])


def fetch_text(url: str, timeout: int = 25) -> str:
    """Fetch a web page and return its visible text content.

    Flow:
        URL (e.g. "https://example.com/article")
            → GET request with custom User-Agent header
            → Receive raw HTML
            → BeautifulSoup parses HTML, strips tags/scripts/styles
            → Extract visible text, collapse whitespace
            → Truncate to first 5000 chars (keeps context window manageable)
            → Return clean plain text

    The 5000-char limit prevents a single large page from overwhelming
    the LLM's context window when injected as research material.

    Args:
        url: full URL to fetch
        timeout: request timeout in seconds (default 25)

    Returns:
        str: plain text content of the page, truncated to 5000 chars

    Raises:
        requests.HTTPError: if the page returns a non-2xx status
        requests.Timeout: if the request exceeds the timeout
    """
    headers = {"User-Agent": "research-agent/1.0"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    return text[:5000]


