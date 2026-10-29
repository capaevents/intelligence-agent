"""AI client with a fallback chain (same idea as the tracker's): when one model fails or runs out of free
quota, the next is tried. Every provider speaks the same OpenAI-style chat API. A provider is only used if its
key is set. The first working model is kept for the whole run so the agent's behaviour stays consistent."""
import os
import time

import requests

CHAIN = [
    ("groq", "https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY", ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]),
    ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY",
     ["gemini-3.8-flash", "gemini-flash-latest"]),
    ("cloudflare", "https://api.cloudflare.com/client/v4/accounts/{account}/ai/v1/chat/completions", "CF_AI_TOKEN",
     ["@cf/meta/llama-3.3-70b-instruct-fp8-fast"]),
    ("openrouter", "https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY",
     ["nvidia/nemotron-3-super-120b-a12b:free"]),
]


class LLMUnavailable(Exception):
    pass


class LLM:
    def __init__(self, env=None, http=None, sleep=time.sleep):
        self.env = env if env is not None else os.environ
        self.http = http or requests.Session()
        self.sleep = sleep
        self.dead = set()
        self.used = []
        self.log = []

    def targets(self):
        out = []
        custom = self.env.get("AGENT_MODELS", "").strip()
        wanted = [s.strip() for s in custom.split(",") if s.strip()] if custom else None
        for provider, url, keyvar, models in CHAIN:
            key = self.env.get(keyvar)
            if not key:
                continue
            if provider == "cloudflare":
                if not self.env.get("CF_ACCOUNT_ID"):
                    continue
                url = url.format(account=self.env["CF_ACCOUNT_ID"])
            for m in models:
                spec = f"{provider}:{m}"
                if wanted is None or spec in wanted:
                    out.append((spec, url, key, m))
        return out

    def chat(self, messages, max_tokens=1500):
        """Try each model in turn. A reasoning model that spends its whole allowance thinking (empty or cut-off
        answer) is retried once with 3x the room; a per-minute rate limit (429) is waited out, not given up on."""
        tried = []
        for spec, url, key, model in self.targets():
            if spec in self.dead:
                continue
            tried.append(spec)
            budget = max_tokens
            for attempt in range(3):
                try:
                    r = self.http.post(url, timeout=90, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                       json={"model": model, "temperature": 0.2, "max_tokens": budget, "messages": messages})
                except requests.RequestException as e:
                    self.log.append(f"{spec}: {type(e).__name__}")
                    self.dead.add(spec)
                    break
                if r.status_code == 200:
                    try:
                        choice = r.json()["choices"][0]
                        text = choice["message"]["content"] or ""
                        finish = choice.get("finish_reason")
                    except (ValueError, KeyError, IndexError, TypeError):
                        self.log.append(f"{spec}: unreadable answer")
                        self.dead.add(spec)
                        break
                    if text.strip() and finish != "length":
                        self.used.append(spec)
                        return text, spec
                    self.log.append(f"{spec}: {'cut off' if text.strip() else 'empty answer'} at {budget} tokens")
                    if attempt == 0:
                        budget = max_tokens * 3
                        continue
                    self.dead.add(spec)
                    break
                if r.status_code == 429 and attempt < 2:
                    wait = _retry_after(r)
                    self.log.append(f"{spec}: HTTP 429, waiting {wait:.0f}s")
                    self.sleep(wait)
                    continue
                if r.status_code >= 500 and attempt == 0:
                    self.sleep(3)
                    continue
                self.log.append(f"{spec}: HTTP {r.status_code}")
                self.dead.add(spec)
                break
        raise LLMUnavailable("no AI model answered (tried: " + (", ".join(tried) or "none - no keys set") + ")")


def _retry_after(r, default=8.0, cap=25.0):
    try:
        return min(float(getattr(r, "headers", {}).get("retry-after", default)), cap)
    except (TypeError, ValueError):
        return default
