"""Run once:  python -m agent            (saves to Supabase if SUPABASE_URL/SUPABASE_SERVICE_KEY are set)
              python -m agent --dry-run  (never touches a database; writes out/run-*.json)"""
import argparse
import json
import os
import sys

from .config import GOAL, LIMITS, allowed_domains
from .llm import LLM
from .loop import run
from .store import DryRunStore, SupabaseStore
from .tools import Tools


def main(argv=None):
    ap = argparse.ArgumentParser(prog="agent")
    ap.add_argument("--dry-run", action="store_true", help="do not read or write Supabase")
    ap.add_argument("--items-file", help="(dry run) JSON list of tracker items to let the agent read")
    ap.add_argument("--steps", type=int, help="override the step limit")
    ap.add_argument("--goal-file", help="use a different goal text")
    a = ap.parse_args(argv)

    limits = dict(LIMITS)
    if a.steps:
        limits["max_steps"] = a.steps
    goal = open(a.goal_file).read() if a.goal_file else GOAL
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    store = DryRunStore(a.items_file) if a.dry_run or not (url and key) else SupabaseStore(url, key)
    print("store:", type(store).__name__)

    llm = LLM()
    result = run(llm, Tools(store, allowed_domains(), limits), goal, limits)
    where = store.save_run(result)
    print(json.dumps({k: result[k] for k in ("stopped", "model", "seconds", "summary")}, indent=2))
    print(f"{len(result['findings'])} suggested finding(s), {len(result['steps'])} step(s) -> {where}")
    return 1 if result["stopped"] in ("ai_unavailable", "bad_output") else 0


if __name__ == "__main__":
    sys.exit(main())
