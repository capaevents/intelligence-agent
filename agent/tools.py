"""The agent's four tools. Three only read; save_finding is the one write, and it only adds a row marked
"suggested". There is deliberately no tool to send email, edit tracker items or touch people's records.
Every tool returns a plain dict and never raises: a problem comes back as {"error": "..."} so the model can adapt."""
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, urlsplit, urlunsplit

import feedparser
import requests
from bs4 import BeautifulSoup

from .config import THEMES

UA = "capa-intelligence-agent/1.0 (+public pages only)"

TOOL_HELP = """Tools (call ONE per turn):
- read_tracker  {"days": 7, "query": ""}  : items CAPA's tracker already collected (title, url, pao, relevance). Read-only.
- search_news   {"query": "..."}          : Google News search, last 30 days. Read-only.
- fetch_page    {"url": "https://..."}    : text and links of one public page. Only allowlisted sites work. Read-only.
- save_finding  {"title": "...", "url": "...", "summary": "...", "reason": "...", "theme": "..."} : save one suggested
  finding for staff to review. url must be one you have already seen in a tool result. theme is one of: %s.
Finish with {"done": true, "summary": "<2-3 sentences on what you found and what you could not check>"}.""" % ", ".join(THEMES)


def norm(url):
    p = urlsplit((url or "").strip())
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/"), p.query, ""))


class Tools:
    def __init__(self, store, allowed_domains, limits, http=None):
        self.store = store
        self.allowed = [d.lower() for d in allowed_domains]
        self.limits = limits
        self.http = http or requests.Session()
        self.http.headers.update({"User-Agent": UA})
        self.seen = set()
        self.findings = []
        self.count = {"search_news": 0, "fetch_page": 0, "save_finding": 0}

    # ---- plumbing ----
    def call(self, name, args):
        fn = {"read_tracker": self.read_tracker, "search_news": self.search_news,
              "fetch_page": self.fetch_page, "save_finding": self.save_finding}.get(name)
        if fn is None:
            return {"error": f"unknown tool {name!r}; use one of read_tracker, search_news, fetch_page, save_finding"}
        if not isinstance(args, dict):
            return {"error": "args must be a JSON object"}
        try:
            return fn(**args)
        except TypeError as e:
            return {"error": f"bad arguments: {e}"}
        except Exception as e:  # noqa: BLE001 - a tool failure must not crash the run
            return {"error": f"{type(e).__name__}: {str(e)[:150]}"}

    def host_allowed(self, url):
        p = urlsplit(url)
        host = (p.hostname or "").lower()
        return p.scheme in ("http", "https") and any(host == d or host.endswith("." + d) for d in self.allowed)

    # ---- tools ----
    def read_tracker(self, days=7, query=""):
        days = max(1, min(int(days), 30))
        rows = self.store.recent_items(days)
        q = (query or "").lower().strip()
        out = []
        for r in rows:
            text = f"{r.get('title', '')} {r.get('pao_id', '')}".lower()
            if q and q not in text:
                continue
            out.append({"title": r.get("title"), "url": r.get("url"), "pao": r.get("pao_id"),
                        "date": r.get("published_at") or r.get("first_seen_at"),
                        "relevance": r.get("relevance") or r.get("ai_relevance")})
            if r.get("url"):
                self.seen.add(norm(r["url"]))
            if len(out) >= 20:
                break
        return {"count": len(out), "items": out}

    def search_news(self, query):
        if self.count["search_news"] >= self.limits["max_searches"]:
            return {"error": "search limit reached for this run; use what you have or fetch a page"}
        self.count["search_news"] += 1
        url = f"https://news.google.com/rss/search?q={quote(str(query) + ' when:30d')}&hl=en&gl=US&ceid=US:en"
        r = self.http.get(url, timeout=25)
        r.raise_for_status()
        feed = feedparser.parse(r.content)
        out = []
        for e in feed.entries[:8]:
            link = e.get("link", "")
            out.append({"title": e.get("title", ""), "url": link,
                        "source": (e.get("source") or {}).get("title", ""), "date": e.get("published", "")})
            self.seen.add(norm(link))
        return {"count": len(out), "results": out}

    def fetch_page(self, url):
        if not self.host_allowed(url):
            return {"error": "that site is not on the allowlist: " + ", ".join(self.allowed)}
        if self.count["fetch_page"] >= self.limits["max_fetches"]:
            return {"error": "fetch limit reached for this run"}
        self.count["fetch_page"] += 1
        r = self.http.get(url, timeout=25)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml") if _has_lxml() else BeautifulSoup(r.text, "html.parser")
        for t in soup(["script", "style", "nav", "footer", "header"]):
            t.decompose()
        text = " ".join(soup.get_text(" ").split())
        links = []
        for a in soup.find_all("a", href=True):
            href = requests.compat.urljoin(r.url, a["href"])
            label = " ".join(a.get_text(" ").split())
            if label and href.startswith("http"):
                links.append({"text": label[:100], "url": href})
                self.seen.add(norm(href))
            if len(links) >= 15:
                break
        self.seen.add(norm(url))
        self.seen.add(norm(r.url))
        return {"url": r.url, "text": text[:2000], "links": links}

    def save_finding(self, title, url, summary, reason, theme="other"):
        if self.count["save_finding"] >= self.limits["max_findings"]:
            return {"error": "finding limit reached"}
        if norm(url) not in self.seen:
            return {"error": "url not seen in any tool result this run; only save sources you actually found"}
        if any(norm(f["url"]) == norm(url) for f in self.findings):
            return {"error": "already saved"}
        title, summary, reason = str(title).strip(), str(summary).strip(), str(reason).strip()
        if not (title and summary and reason):
            return {"error": "title, summary and reason are all required"}
        self.count["save_finding"] += 1
        self.findings.append({"title": title[:300], "url": url.strip(), "summary": summary[:600], "reason": reason[:300],
                              "theme": theme if theme in THEMES else "other", "status": "suggested",
                              "found_at": datetime.now(timezone.utc).isoformat()})
        return {"saved": True, "total": len(self.findings)}


def _has_lxml():
    try:
        import lxml  # noqa: F401
        return True
    except ImportError:
        return False


def clip(result, limit):
    s = json.dumps(result, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + '..."(truncated)"'
