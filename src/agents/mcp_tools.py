"""
MCP Tools for Design Agent.

Uses Model Context Protocol to connect to external services.
Currently implements: Brave Web Search
"""

import os
import asyncio
from typing import Optional
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

import httpx


load_dotenv()


# =============================================================================
# Brave Search Tool (MCP-style implementation)
# =============================================================================

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


async def brave_search_async(query: str, num_results: int = 5) -> list[dict]:
    """
    Search the web using Brave Search API.
    
    Returns list of results with title, url, and description.
    """
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise ValueError("BRAVE_API_KEY not found in environment")
    
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    
    params = {
        "q": query,
        "count": num_results
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            BRAVE_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
    
    results = []
    web_results = data.get("web", {}).get("results", [])
    
    for item in web_results[:num_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "description": item.get("description", "")
        })
    
    return results


def brave_search(query: str, num_results: int = 5) -> list[dict]:
    """Sync wrapper for brave_search_async."""
    return asyncio.run(brave_search_async(query, num_results))


# =============================================================================
# Research Tool (Search + Summarize)
# =============================================================================

class ResearchTool:
    """
    Tool that searches the web and summarizes findings.
    
    Uses Brave Search for web search and Groq LLM for summarization.
    This keeps the context size manageable.
    """
    
    def __init__(
        self,
        model: str = "llama3.1",
        temperature: float = 0
    ):
        
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
        )
    
    def search_and_summarize(
        self,
        query: str,
        focus: str = "",
        num_results: int = 5
    ) -> str:
        """
        Search the web and return a summarized result.
        
        Args:
            query: Search query
            focus: What aspect to focus on in summary (optional)
            num_results: Number of search results to process
            
        Returns:
            Summarized findings (concise, ~200-300 words)
        """
        # Step 1: Search
        try:
            results = brave_search(query, num_results)
        except Exception as e:
            return f"Search failed: {str(e)}"
        
        if not results:
            return "No search results found."
        
        # Step 2: Format results for summarization
        results_text = "\n\n".join([
            f"**{r['title']}**\n{r['url']}\n{r['description']}"
            for r in results
        ])
        
        # Step 3: Summarize
        focus_instruction = f"\nFocus specifically on: {focus}" if focus else ""
        
        messages = [
            SystemMessage(content=f"""You are a research assistant. Summarize the search results into key findings.

Be concise (200-300 words max). Focus on:
- Factual information relevant to AI/telecom system design
- Technical specifications or standards mentioned
- Architectural patterns or best practices
{focus_instruction}

Format as bullet points for easy reading."""),
            HumanMessage(content=f"""Search query: {query}

Search results:
{results_text}

Provide a concise summary of key findings:""")
        ]
        
        response = self.llm.invoke(messages)
        return response.content


# =============================================================================
# LangChain Tool Wrapper
# =============================================================================

# Global research tool instance
_research_tool: Optional[ResearchTool] = None


def get_research_tool() -> ResearchTool:
    """Get or create the research tool singleton."""
    global _research_tool
    if _research_tool is None:
        _research_tool = ResearchTool()
    return _research_tool


@tool
def research(query: str, focus: str = "") -> str:
    """
    Search the web and get summarized findings.
    
    Use this tool to research:
    - Latest 3GPP/ETSI specifications
    - AI agent architectural patterns
    - Telecom network integration approaches
    - Multi-agent system best practices
    
    Args:
        query: What to search for (be specific)
        focus: Optional focus area for the summary
        
    Returns:
        Summarized key findings from web search
    """
    tool_instance = get_research_tool()
    return tool_instance.search_and_summarize(query, focus)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing MCP Tools...\n")
    
    # Test 1: Raw Brave Search
    print("=" * 60)
    print("TEST 1: Brave Search (raw)")
    print("=" * 60)
    
    try:
        results = brave_search("3GPP AI agent network architecture", num_results=3)
        print(f"Found {len(results)} results:\n")
        for r in results:
            print(f"  • {r['title']}")
            print(f"    {r['url']}")
            print(f"    {r['description'][:100]}...\n")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Research Tool (Search + Summarize)
    print("\n" + "=" * 60)
    print("TEST 2: Research Tool (search + summarize)")
    print("=" * 60)
    
    try:
        result = research.invoke({
            "query": "3GPP 5G core network AI agent integration NWDAF",
            "focus": "architectural patterns for AI agent integration"
        })
        print(f"\nSummarized findings:\n{result}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✓ MCP Tools test complete")