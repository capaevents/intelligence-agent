"""AI client with a fallback chain (same idea as the tracker's): when one model fails or runs out of free
quota, the next is tried. Every provider speaks the same OpenAI-style chat API. A provider is only used if its
key is set. The first working model is kept for the whole run so the agent's behaviour stays consistent."""
import os

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
    def __init__(self, env=None, http=None):
        self.env = env if env is not None else os.environ
        self.http = http or requests.Session()
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

    def chat(self, messages, max_tokens=900):
        tried = []
        for spec, url, key, model in self.targets():
            if spec in self.dead:
                continue
            tried.append(spec)
            try:
                r = self.http.post(url, timeout=90, headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                                   json={"model": model, "temperature": 0.2, "max_tokens": max_tokens, "messages": messages})
            except requests.RequestException as e:
                self.log.append(f"{spec}: {type(e).__name__}")
                self.dead.add(spec)
                continue
            if r.status_code == 200:
                try:
                    text = r.json()["choices"][0]["message"]["content"] or ""
                except (ValueError, KeyError, IndexError, TypeError):
                    self.log.append(f"{spec}: unreadable answer")
                    self.dead.add(spec)
                    continue
                if text.strip():
                    self.used.append(spec)
                    return text, spec
                self.log.append(f"{spec}: empty answer")
                self.dead.add(spec)
                continue
            self.log.append(f"{spec}: HTTP {r.status_code}")
            self.dead.add(spec)
        raise LLMUnavailable("no AI model answered (tried: " + (", ".join(tried) or "none - no keys set") + ")")
