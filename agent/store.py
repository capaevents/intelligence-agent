"""Where the agent reads the tracker's items and saves its results.
SupabaseStore: normal. DryRunStore: no database at all; results go to a local JSON file in out/."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests


class DryRunStore:
    def __init__(self, items_file=None, out_dir="out"):
        self.items = json.loads(Path(items_file).read_text()) if items_file else []
        self.out_dir = Path(out_dir)

    def recent_items(self, days):
        return self.items

    def save_run(self, result):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return str(path)


class SupabaseStore:
    """Supabase REST API. The key stays server side (a GitHub secret). Read: items. Write: agent_runs, agent_findings."""

    def __init__(self, url, key):
        self.base = url.rstrip("/") + "/rest/v1"
        self.headers = {"apikey": key, "Content-Type": "application/json", "User-Agent": "capa-intelligence-agent/1.0"}
        if key.startswith("eyJ"):
            self.headers["Authorization"] = f"Bearer {key}"

    def _req(self, method, path, **kw):
        r = requests.request(method, self.base + path, headers={**self.headers, **kw.pop("headers", {})}, timeout=60, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Supabase {method} {path.split('?')[0]} failed {r.status_code}: {r.text[:200]}")
        return r

    def recent_items(self, days):
        since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        cols = "id,pao_id,title,url,published_at,first_seen_at,relevance,ai_relevance,kind"
        path = (f"/items?select={cols}&first_seen_at=gte.{since}&removed_at=is.null&is_baseline=is.false"
                f"&order=first_seen_at.desc&limit=200")
        try:
            return self._req("GET", path).json()
        except RuntimeError:  # older database without the AI columns: fall back to the base set
            return self._req("GET", path.replace(",ai_relevance", "")).json()

    def save_run(self, result):
        run = self._req("POST", "/agent_runs", headers={"Prefer": "return=representation"}, json={
            "stopped": result["stopped"], "model": result["model"], "seconds": result["seconds"],
            "summary": result["summary"], "steps": result["steps"]}).json()[0]
        if result["findings"]:
            rows = [{**f, "run_id": run["id"]} for f in result["findings"]]
            for r in rows:
                r.pop("found_at", None)
            self._req("POST", "/agent_findings", json=rows)
        return f"agent_runs #{run['id']} ({len(result['findings'])} findings)"
