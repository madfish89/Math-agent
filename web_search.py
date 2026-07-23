"""Web search integration for looking up mathematical theorems and proofs.

Uses Wolfram Alpha (via free API), Wikipedia/MathWorld lookups, and
general web search to cite external sources.
"""
import json
import urllib.request
import urllib.parse
import re
import os

from config import BASE_DIR

WOLFRAM_APP_ID = os.environ.get("WOLFRAM_APP_ID", "")
WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary"
MATHWORLD_SEARCH = "https://mathworld.wolfram.com/search/?query="


def search_wolfram(query):
    """Query Wolfram Alpha for a mathematical answer."""
    if not WOLFRAM_APP_ID:
        return {"available": False, "error": "No WOLFRAM_APP_ID set. Get a free ID at https://developer.wolframalpha.com/portal"}

    encoded = urllib.parse.quote(query)
    url = f"http://api.wolframalpha.com/v2/query?input={encoded}&appid={WOLFRAM_APP_ID}&format=plaintext&output=json"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MathAgent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        pods = []
        if "queryresult" in data and "pods" in data["queryresult"]:
            for pod in data["queryresult"]["pods"]:
                title = pod.get("title", "")
                subpods = pod.get("subpods", [])
                text = " ".join(sp.get("plaintext", "") for sp in subpods if sp.get("plaintext"))
                if text.strip():
                    pods.append({"title": title, "text": text[:500]})

        return {
            "available": True,
            "source": "Wolfram Alpha",
            "pods": pods[:5],
            "url": f"https://www.wolframalpha.com/input?i={encoded}"
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def search_wikipedia(query):
    """Search Wikipedia for a mathematical concept/theorem."""
    # Try the summary API first
    title = query.replace(" ", "_")
    title = re.sub(r'[^a-zA-Z0-9_]', '', title)
    url = f"{WIKIPEDIA_API}/{title}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MathAgent/1.0 (educational)"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        return {
            "available": True,
            "source": "Wikipedia",
            "title": data.get("title", ""),
            "extract": data.get("extract", "")[:600],
            "url": data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{title}")
        }
    except urllib.error.HTTPError:
        # Try search instead
        return _wikipedia_search(query)
    except Exception as e:
        return {"available": False, "error": str(e)}


def _wikipedia_search(query):
    """Fallback: use Wikipedia's search API."""
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": f"mathematics {query}",
        "srlimit": 3,
        "format": "json",
        "origin": "*"
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MathAgent/1.0 (educational)"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        results = []
        for item in data.get("query", {}).get("search", [])[:3]:
            snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
            results.append({
                "title": item.get("title", ""),
                "snippet": snippet[:300],
                "url": f"https://en.wikipedia.org/wiki/{item.get('title', '').replace(' ', '_')}"
            })

        return {
            "available": True,
            "source": "Wikipedia Search",
            "results": results
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def search_mathworld(query):
    """Generate a MathWorld search URL (no API, but useful for citations)."""
    encoded = urllib.parse.quote(query)
    return {
        "available": True,
        "source": "Wolfram MathWorld",
        "url": f"https://mathworld.wolfram.com/search/?query={encoded}",
        "note": "MathWorld is an encyclopedic reference. Open the URL for detailed definitions and proofs."
    }


def lookup_theorem(theorem_name):
    """Look up a theorem across multiple sources."""
    results = {}

    # Wikipedia
    wiki = search_wikipedia(theorem_name)
    if wiki.get("available"):
        results["wikipedia"] = wiki

    # MathWorld
    results["mathworld"] = search_mathworld(theorem_name)

    # Wolfram Alpha (if API key available)
    wa = search_wolfram(theorem_name)
    if wa.get("available"):
        results["wolfram_alpha"] = wa

    return results


def search_math_concept(query):
    """General-purpose math concept search. Tries all sources."""
    results = {}

    # Try Wikipedia first (most reliable, no API key needed)
    wiki = search_wikipedia(query)
    if wiki.get("available"):
        results["wikipedia"] = wiki

    # MathWorld URL
    results["mathworld"] = search_mathworld(query)

    # Wolfram Alpha
    wa = search_wolfram(query)
    if wa.get("available"):
        results["wolfram_alpha"] = wa

    return results


def format_sources(sources):
    """Format source results for display."""
    lines = []
    for source_name, data in sources.items():
        if not data.get("available", True):
            continue
        lines.append(f"\n  [{source_name}]")
        if "title" in data:
            lines.append(f"  Title: {data['title']}")
        if "extract" in data:
            lines.append(f"  Summary: {data['extract'][:200]}...")
        if "url" in data:
            lines.append(f"  URL: {data['url']}")
        if "pods" in data:
            for pod in data["pods"][:2]:
                lines.append(f"  {pod['title']}: {pod['text'][:150]}")
        if "results" in data:
            for r in data["results"][:2]:
                lines.append(f"  {r['title']}: {r['snippet'][:150]}")
    return "\n".join(lines) if lines else "  No external sources found."


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "Fundamental Theorem of Calculus"
    print(f"Searching for: {query}\n")
    sources = search_math_concept(query)
    print(format_sources(sources))