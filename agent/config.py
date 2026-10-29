"""Everything you might want to change without touching the logic."""
import os

# The only sites fetch_page may open (the site itself or any of its subdomains).
DEFAULT_ALLOWED_DOMAINS = ["iaasb.org", "ethicsboard.org", "ifac.org", "ifrs.org"]

THEMES = ["ai_digitalisation", "talent", "sustainability", "ethics_standards", "other"]

# Hard limits for one run. The loop stops at the first one it reaches.
LIMITS = {
    "max_steps": 12,        # model turns
    "max_searches": 3,      # search_news calls
    "max_fetches": 4,       # fetch_page calls
    "max_findings": 10,     # save_finding calls
    "max_seconds": 480,     # wall-clock time
    "max_tool_chars": 2500, # size of any tool result shown to the model
}

GOAL = """Find developments from the last 30 days that CAPA (the Confederation of Asian and Pacific \
Accountants) should know about, on these themes: AI and digitalisation; talent attraction and retention; \
sustainability reporting and assurance; ethics and professional standards. Priority sources are the \
IAASB, IESBA, IFAC and IFRS/ISSB, and what member accountancy bodies (PAOs) in Asia-Pacific have said or done.
Use read_tracker first to see what CAPA already holds, so you do not repeat it. Then search and fetch to find \
what is missing. Save each useful finding with save_finding, including the source URL and one sentence on why \
it matters to CAPA. Quality over quantity: it is fine to save nothing if nothing new is worth flagging."""


def allowed_domains(env=None):
    env = env if env is not None else os.environ
    raw = env.get("AGENT_ALLOWED_DOMAINS", "").strip()
    return [d.strip().lower() for d in raw.split(",") if d.strip()] if raw else list(DEFAULT_ALLOWED_DOMAINS)
