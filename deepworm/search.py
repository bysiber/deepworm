"""Web search with multiple providers.

Supports DuckDuckGo (default), Brave Search API, SearXNG, and Tavily.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote_plus, unquote

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    body: Optional[str] = None


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
}


def search_web(
    query: str,
    max_results: int = 8,
    region: str = "wt-wt",
    cache: "Cache | None" = None,
    provider: str = "duckduckgo",
) -> list[SearchResult]:
    """Search the web using the specified provider.

    Providers: duckduckgo (default), brave, searxng, tavily
    """
    # Check cache first
    if cache is not None:
        cached = cache.get("search", query)
        if cached is not None:
            logger.debug("Search cache hit: %s", query[:60])
            return [SearchResult(**r) for r in cached]

    results: list[SearchResult] = []

    if provider == "brave":
        try:
            results = _search_brave(query, max_results)
        except Exception as e:
            logger.debug("Brave search failed: %s", e)
    elif provider == "searxng":
        try:
            results = _search_searxng(query, max_results)
        except Exception as e:
            logger.debug("SearXNG search failed: %s", e)
    elif provider == "tavily":
        try:
            results = _search_tavily(query, max_results)
        except Exception as e:
            logger.debug("Tavily search failed: %s", e)

    # Default/fallback: DuckDuckGo
    if not results:
        try:
            results = _search_ddgs(query, max_results)
        except Exception:
            pass

    if not results:
        # Fallback: DuckDuckGo HTML scraping
        try:
            results = _search_html(query, max_results)
        except Exception:
            pass

    results = results[:max_results]

    # Store in cache
    if cache is not None and results:
        cache.set("search", query, [
            {"title": r.title, "url": r.url, "snippet": r.snippet}
            for r in results
        ])

    return results


def _search_ddgs(query: str, max_results: int) -> list[SearchResult]:
    """Try the ddgs/duckduckgo_search library."""
    results = []
    try:
        from ddgs import DDGS
        ddgs = DDGS()
        for r in ddgs.text(query, max_results=max_results):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
            ))
    except ImportError:
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                    ))
        except (ImportError, Exception):
            raise
    return results


def _search_html(query: str, max_results: int) -> list[SearchResult]:
    """Fallback: scrape DuckDuckGo HTML results."""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    resp = httpx.get(url, headers=_HEADERS, timeout=15.0, follow_redirects=True)
    resp.raise_for_status()

    results = []
    # Parse result blocks
    blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]+href="([^"]*)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        resp.text,
        re.DOTALL,
    )
    for href, title, snippet in blocks[:max_results]:
        # DuckDuckGo wraps URLs in a redirect
        actual_url = _extract_ddg_url(href)
        clean_title = re.sub(r'<[^>]+>', '', title).strip()
        clean_snippet = re.sub(r'<[^>]+>', '', snippet).strip()
        if actual_url and clean_title:
            results.append(SearchResult(
                title=clean_title,
                url=actual_url,
                snippet=clean_snippet,
            ))

    return results


def _extract_ddg_url(href: str) -> str:
    """Extract real URL from DuckDuckGo redirect URL."""
    if "duckduckgo.com" in href and "uddg=" in href:
        match = re.search(r'uddg=([^&]+)', href)
        if match:
            return unquote(match.group(1))
    return href


def fetch_page_text(url: str, timeout: float = 10.0, cache: "Cache | None" = None) -> str:
    """Fetch and extract text content from a URL."""
    # Skip non-text URLs
    skip_extensions = ('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.mp4', '.mp3', '.zip', '.tar', '.gz')
    if any(url.lower().endswith(ext) for ext in skip_extensions):
        return ""

    # Check cache first
    if cache is not None:
        cached = cache.get("page", url)
        if cached is not None:
            logger.debug("Page cache hit: %s", url[:80])
            return cached

    try:
        resp = httpx.get(url, headers=_HEADERS, timeout=timeout, follow_redirects=True)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return ""

        text = resp.text
        text = _extract_text_from_html(text)
        # Truncate to reasonable length
        if len(text) > 8000:
            text = text[:8000]

        # Store in cache
        if cache is not None and text:
            cache.set("page", url, text)

        return text
    except Exception:
        return ""


def _extract_text_from_html(html: str) -> str:
    """Extract readable text from HTML without external dependencies."""

    # Remove elements that don't contain useful content
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<head[^>]*>.*?</head>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<nav[^>]*>.*?</nav>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<footer[^>]*>.*?</footer>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<header[^>]*>.*?</header>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<aside[^>]*>.*?</aside>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)

    # Add line breaks for block elements
    for tag in ['p', 'br', 'div', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'tr']:
        text = re.sub(rf'</?{tag}[^>]*>', '\n', text, flags=re.IGNORECASE)

    # Remove remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Decode HTML entities
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    text = text.replace('&nbsp;', ' ')
    text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
    text = re.sub(r'&#x([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), text)

    # Clean up whitespace
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]  # remove empty lines
    text = '\n'.join(lines)

    # Collapse multiple spaces within lines
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()


def _search_brave(query: str, max_results: int) -> list[SearchResult]:
    """Search using Brave Search API. Requires BRAVE_API_KEY env var."""
    import os

    api_key = os.getenv("BRAVE_API_KEY", "")
    if not api_key:
        raise ValueError("BRAVE_API_KEY not set")

    resp = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": max_results},
        headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
        timeout=15.0,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("web", {}).get("results", [])[:max_results]:
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("description", ""),
        ))
    return results


def _search_searxng(query: str, max_results: int) -> list[SearchResult]:
    """Search using a SearXNG instance. Requires SEARXNG_URL env var."""
    import os

    base_url = os.getenv("SEARXNG_URL", "")
    if not base_url:
        raise ValueError("SEARXNG_URL not set")

    resp = httpx.get(
        f"{base_url.rstrip('/')}/search",
        params={"q": query, "format": "json", "pageno": 1},
        headers=_HEADERS,
        timeout=15.0,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("results", [])[:max_results]:
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("content", ""),
        ))
    return results


def _search_tavily(query: str, max_results: int) -> list[SearchResult]:
    """Search using Tavily API. Requires TAVILY_API_KEY env var."""
    import os

    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not set")

    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_raw_content=True,
    )

    results = []
    for item in response.get("results", [])[:max_results]:
        results.append(SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("content", ""),
            body=item.get("raw_content", None),
        ))
    return results
