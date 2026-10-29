"""The agent loop. Each turn the model names ONE tool (or says it is done); the code runs it and feeds the result
back. The code, not the model, enforces every limit, so a confused model cannot run away."""
import json
import re
import time

from .llm import LLMUnavailable
from .tools import TOOL_HELP, clip

SYSTEM = """You are a research agent for CAPA, the Confederation of Asian and Pacific Accountants. You help staff by \
finding developments they should know about. You never invent sources: every finding needs a URL you have seen in a \
tool result. You cannot send anything or change any records. Staff review everything you save.

Reply with ONLY a single JSON object each turn, no other text: {"tool": "<name>", "args": {...}} to use a tool, or \
{"done": true, "summary": "..."} when finished. Be efficient: you have a small budget of steps."""


def parse_action(text):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    for cand in (t, t[t.find("{"):t.rfind("}") + 1] if "{" in t else ""):
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except ValueError:
            continue
    return None


def run(llm, tools, goal, limits, clock=time.monotonic):
    start = clock()
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"GOAL:\n{goal}\n\n{TOOL_HELP}\n\nLimits: {limits['max_steps']} steps, "
                                            f"{limits['max_searches']} searches, {limits['max_fetches']} page fetches."}]
    steps, summary, stopped, bad = [], "", "max_steps", 0
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
            steps.append({"step": n, "tool": None, "ok": False, "note": "unparseable answer"})
            if bad >= 2:
                stopped = "bad_output"
                break
            messages += [{"role": "assistant", "content": text[:500]},
                         {"role": "user", "content": 'Reply with ONLY one JSON object, e.g. {"tool": "read_tracker", "args": {}}.'}]
            continue
        bad = 0
        if action.get("done"):
            summary = str(action.get("summary", ""))[:800]
            steps.append({"step": n, "tool": "done", "ok": True})
            stopped = "done"
            break
        name, args = action.get("tool"), action.get("args", {})
        result = tools.call(name, args)
        steps.append({"step": n, "tool": name, "args": args, "ok": "error" not in result})
        messages += [{"role": "assistant", "content": json.dumps(action)},
                     {"role": "user", "content": f"RESULT of {name}: " + clip(result, limits["max_tool_chars"])}]
    return {"findings": tools.findings, "steps": steps, "summary": summary, "stopped": stopped,
            "model": (llm.used[-1] if llm.used else None), "seconds": round(clock() - start, 1), "ai_log": llm.log}
