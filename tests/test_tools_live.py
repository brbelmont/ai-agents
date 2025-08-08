from utils.settings import settings
import httpx


def test_tavily():
    print("\n=== Tavily Search Test ===")
    headers = {
        "Authorization": f"Bearer {settings.tavily_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": "retrieval-augmented generation best practices",
        "max_results": 3,
    }
    with httpx.Client(timeout=30) as client:
        r = client.post("https://api.tavily.com/search", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

    results = data.get("results", [])
    if not results:
        print("No results returned — check API key or query.")
    else:
        for item in results:
            print(f"- {item.get('title')} — {item.get('url')}")


def test_firecrawl():
    print("\n=== Firecrawl Scrape Test ===")
    url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    headers = {
        "Authorization": f"Bearer {settings.firecrawl_api_key}",
        "Content-Type": "application/json",
    }
    payload = {"url": url, "onlyMainContent": True, "formats": ["markdown"]}
    with httpx.Client(timeout=60) as client:
        r = client.post(
            "https://api.firecrawl.dev/v1/scrape", headers=headers, json=payload
        )
        if r.status_code != 200:
            print("Firecrawl error:", r.text)
            r.raise_for_status()
        j = r.json()

    data = j.get("data") or {}
    content = data.get("markdown") or data.get("html") or data.get("rawHtml") or ""
    print(f"Scraped content length: {len(content)}")
    print(f"First 300 chars:\n{content[:300]}")


if __name__ == "__main__":
    test_tavily()
    test_firecrawl()
