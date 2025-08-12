from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from utils.settings import settings
import httpx
import json


class AgentState(TypedDict, total=False):
    question: str
    findings: List[Dict[str, Any]]
    final_answer: str


class WebFinding(BaseModel):
    title: str
    url: str
    snippet: str


def tavily_search(query: str, k: int = 5) -> List[WebFinding]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.tavily_api_key}",
    }
    payload = {"query": query, "max_results": k}
    with httpx.Client(timeout=30) as client:
        r = client.post("https://api.tavily.com/search", headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    return [
        WebFinding(
            title=i.get("title", ""),
            url=i.get("url", ""),
            snippet=(i.get("content", "")[:300] or ""),
        )
        for i in data.get("results", [])
    ]


def firecrawl_get_markdown(url: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.firecrawl_api_key}",
    }
    payload = {
        "url": url,
        "onlyMainContent": True,  # optionsl: cleaner content
        "formats": ["markdown"],  # ensure markdown is returned
    }
    with httpx.Client(timeout=60) as client:
        r = client.post(
            "https://api.firecrawl.dev/v1/scrape", headers=headers, json=payload
        )
        r.raise_for_status()
        j = r.json()
    # New: pull from data.markdown; fall back to html/rawHtml if needed
    data = j.get("data") or {}
    return data.get("markdown") or data.get("html") or data.get("rawHtml") or ""


def call_llm(system: str, user: str, model: Optional[str] = None) -> str:
    m = model or settings.openai_model
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.openai_api_key}",
    }
    payload = {
        "model": m,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    with httpx.Client(timeout=60) as client:
        r = client.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"]


def plan_node(state: AgentState) -> AgentState:
    system = (
        "You are a precise research planner. Return 3 focused web queries as JSON list."
    )
    user = f"Question: {state['question']}\nReturn JSON list of 3 queries."
    out = call_llm(system, user)
    queries = json.loads(out) if out.strip().startswith("[") else [state["question"]]
    findings = []
    for q in queries:
        for f in tavily_search(q, k=3):
            findings.append(f.model_dump())
    state["findings"] = findings
    return state


def gather_node(state: AgentState) -> AgentState:
    top = state.get("findings", [])[:3]
    docs = []
    for item in top:
        try:
            md = firecrawl_get_markdown(item["url"])
            docs.append(md[:5000])
        except Exception:
            pass
    context = "\n\n---\n\n".join(docs)
    question = state["question"]
    system = "You are a careful research assistant. Use ONLY given context to answer with citations."
    user = f"Question: {question}\n\nContext:\n{context}\n\nFormat: Answer first, then bullet list of sources with titles and URLs."
    state["final_answer"] = call_llm(system, user)
    return state


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("gather", gather_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "gather")
    g.add_edge("gather", END)
    return g.compile()


def main():
    graph = build_graph()
    q = "What are 5 ChatGPT prompt engineering best practices?"
    out = graph.invoke({"question": f"{q}"})
    print(f"Question: {q}\n\n")
    print(out["final_answer"])


if __name__ == "__main__":
    main()
