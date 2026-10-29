"""The agent loop. Each turn the model names ONE tool (or says it is done); the code runs it and feeds the result
back. The code, not the model, enforces every limit, so a confused model cannot run away."""
import json
import re
import time
from datetime import datetime, timezone

from .llm import LLMUnavailable
from .tools import TOOL_HELP, clip

SYSTEM = """You are a research agent for CAPA, the Confederation of Asian and Pacific Accountants. You help staff by \
finding developments they should know about. You never invent sources: every finding needs a URL you have seen in a \
tool result. You cannot send anything or change any records. Staff review everything you save.

Reply with ONLY a single JSON object each turn, no other text: {"tool": "<name>", "args": {...}} to use a tool, or \
{"done": true, "summary": "..."} when finished. Be efficient: you have a small budget of steps. \
A search result (title, url, source, date) is enough to save a finding, so save relevant ones as you go rather than \
fetching more pages first. Never repeat a call that already failed."""


def parse_action(text):
    """The model's chosen action: the last JSON object in its answer that has a "tool" or "done" key. Tolerates
    code fences, <think> blocks and chatter around the JSON."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    dec, found, i = json.JSONDecoder(), None, 0
    while True:
        i = t.find("{", i)
        if i < 0:
            return found
        try:
            obj, end = dec.raw_decode(t, i)
        except ValueError:
            i += 1
            continue
        if isinstance(obj, dict) and ("tool" in obj or "done" in obj):
            found = obj
        i = end


_RESERVED = ("tool", "name", "args", "arguments", "parameters", "input", "done")


def _name_and_args(action):
    """Models sometimes write the arguments beside the tool name instead of inside "args"; accept both."""
    args = next((action[k] for k in ("args", "arguments", "parameters", "input") if isinstance(action.get(k), dict) and action[k]), None)
    if args is None:
        args = {k: v for k, v in action.items() if k not in _RESERVED}
    return action.get("tool") or action.get("name"), args


def run(llm, tools, goal, limits, clock=time.monotonic):
    start = clock()
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Today's date: {datetime.now(timezone.utc).date().isoformat()}.\n\nGOAL:\n{goal}\n\n{TOOL_HELP}\n\nLimits: {limits['max_steps']} steps, "
                                            f"{limits['max_searches']} searches, {limits['max_fetches']} page fetches."}]
    steps, summary, stopped, bad = [], "", "max_steps", 0
    failed, too_early, nudged_empty = set(), 0, False
    for n in range(1, limits["max_steps"] + 1):
        if clock() - start > limits["max_seconds"]:
            stopped = "time_limit"
            break
        try:
            text, _ = llm.chat(messages)
        except LLMUnavailable as e:
            stopped = "ai_unavailable"
            summary = str(e)
            break
        action = parse_action(text)
        if action is None:
            bad += 1
            steps.append({"step": n, "tool": None, "ok": False, "error": "unparseable answer: " + text.strip()[:120]})
            if bad >= 2:
                stopped = "bad_output"
                break
            messages += [{"role": "assistant", "content": text[:500]},
                         {"role": "user", "content": 'Reply with ONLY one JSON object, e.g. {"tool": "read_tracker", "args": {}}.'}]
            continue
        bad = 0
        if action.get("done") and tools.count["search_news"] < limits.get("min_searches", 0) and too_early < 2 \
                and n < limits["max_steps"] - 1:
            too_early += 1
            steps.append({"step": n, "tool": "done", "ok": False, "error": "finished before searching"})
            messages += [{"role": "assistant", "content": json.dumps(action)},
                         {"role": "user", "content": f"Not yet: you have run {tools.count['search_news']} searches and must run at "
                                                     f"least {limits['min_searches']} before finishing. Use search_news now."}]
            continue
        if action.get("done") and not tools.findings and not nudged_empty and n < limits["max_steps"] - 1:
            nudged_empty = True     # one push: finishing empty-handed is allowed, but only after a second look
            steps.append({"step": n, "tool": "done", "ok": False, "error": "finished with nothing saved; asked to look again"})
            messages += [{"role": "assistant", "content": json.dumps(action)},
                         {"role": "user", "content": "You have saved nothing. Look again at the search results: any relevant news "
                                                     "result can be saved by its id with save_finding (the site allowlist only "
                                                     "limits fetch_page, not saving). Save the genuinely relevant ones, or call "
                                                     "done again if truly nothing is relevant."}]
            continue
        if action.get("done"):
            summary = str(action.get("summary", ""))[:800]
            steps.append({"step": n, "tool": "done", "ok": True})
            stopped = "done"
            break
        name, args = _name_and_args(action)
        key = json.dumps([name, args], sort_keys=True, default=str)
        if key in failed:
            result = {"error": "you already tried exactly this and it failed; do something different (arguments go inside \"args\")"}
        else:
            result = tools.call(name, args)
            if "error" in result:
                failed.add(key)
        steps.append({"step": n, "tool": name, "args": args, "ok": "error" not in result,
                      **({"error": str(result["error"])[:200]} if "error" in result else {})})
        left = limits["max_steps"] - n
        hint = f" Steps left: {left}."
        if left <= 3:
            hint += " Save your findings now with save_finding, then finish with done."
        messages += [{"role": "assistant", "content": json.dumps(action)},
                     {"role": "user", "content": f"RESULT of {name}: " + clip(result, limits["max_tool_chars"]) + hint}]
    return {"findings": tools.findings, "steps": steps, "summary": summary, "stopped": stopped,
            "model": (llm.used[-1] if llm.used else None), "seconds": round(clock() - start, 1), "ai_log": llm.log,
            "searches": tools.count["search_news"], "fetches": tools.count["fetch_page"]}
