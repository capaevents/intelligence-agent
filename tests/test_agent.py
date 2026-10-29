"""The agent's safety rules, tested with a scripted fake model and fake web. No network, no AI keys needed."""
import json

from agent.config import LIMITS
from agent.llm import LLM, LLMUnavailable
from agent.loop import parse_action, run
from agent.store import DryRunStore
from agent.tools import Tools

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>x</title>
<item><title>IAASB issues AI exposure draft</title><link>https://news.example.org/a1</link><pubDate>Mon, 26 Oct 2026 01:00:00 GMT</pubDate></item>
</channel></rss>"""


class Resp:
    def __init__(self, content=b"", status=200, url="https://x"):
        self.content, self.status_code, self.url = content, status, url
        self.text = content.decode("utf8", "ignore")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return json.loads(self.text)


class FakeHTTP:
    def __init__(self, resp=None):
        self.headers, self.calls, self.resp = {}, [], resp or Resp(RSS)

    def get(self, url, **kw):
        self.calls.append(url)
        return self.resp


class FakeLLM:
    def __init__(self, script):
        self.script, self.i, self.used, self.log = list(script), 0, [], []

    def chat(self, messages, max_tokens=900):
        text = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        self.used.append("fake:model")
        return (text if isinstance(text, str) else json.dumps(text)), "fake:model"


def tools(http=None, items=None, limits=None):
    store = DryRunStore()
    store.items = items or []
    return Tools(store, ["iaasb.org"], limits or dict(LIMITS), http=http or FakeHTTP())


def act(tool, **args):
    return {"tool": tool, "args": args}


def test_full_run_saves_a_finding():
    t = tools()
    llm = FakeLLM([act("read_tracker"), act("search_news", query="IAASB AI"),
                   act("save_finding", title="IAASB AI exposure draft", url="https://news.example.org/a1",
                       summary="IAASB proposes AI guidance.", reason="Affects auditors in the region.", theme="ai_digitalisation"),
                   {"done": True, "summary": "One finding."}])
    r = run(llm, t, "goal", dict(LIMITS, min_searches=1))
    assert r["stopped"] == "done"
    assert [s["tool"] for s in r["steps"]] == ["read_tracker", "search_news", "save_finding", "done"]
    assert r["findings"][0]["status"] == "suggested" and r["findings"][0]["theme"] == "ai_digitalisation"


def test_cannot_save_a_url_it_never_saw():
    t = tools()
    out = t.call("save_finding", {"title": "t", "url": "https://made.up/x", "summary": "s", "reason": "r"})
    assert "error" in out and not t.findings


def test_fetch_refuses_sites_off_the_allowlist():
    http = FakeHTTP()
    t = tools(http)
    for bad in ("https://evil.com/", "https://evil-iaasb.org/", "https://iaasb.org.evil.com/", "file:///etc/passwd"):
        assert "error" in t.call("fetch_page", {"url": bad})
    assert http.calls == []
    assert t.host_allowed("https://www.iaasb.org/x") and t.host_allowed("https://iaasb.org")


def test_step_limit_is_enforced_by_the_code():
    limits = dict(LIMITS, max_steps=3)
    llm = FakeLLM([act("read_tracker")])      # never says done
    r = run(llm, tools(limits=limits), "goal", limits)
    assert r["stopped"] == "max_steps" and len(r["steps"]) == 3


def test_search_limit():
    t = tools(limits=dict(LIMITS, max_searches=1))
    assert "results" in t.call("search_news", {"query": "a"})
    assert "error" in t.call("search_news", {"query": "b"})


def test_bad_output_stops_the_run():
    r = run(FakeLLM(["I think we should search"]), tools(), "goal", dict(LIMITS))
    assert r["stopped"] == "bad_output"


def test_unknown_tool_and_bad_args_do_not_crash():
    t = tools()
    assert "error" in t.call("send_email", {"to": "x"})
    assert "error" in t.call("read_tracker", {"nonsense": 1})
    assert "error" in t.call("read_tracker", "not a dict")


def test_no_tool_can_write_except_save_finding():
    assert {n for n in dir(Tools) if not n.startswith("_")} >= {"read_tracker", "search_news", "fetch_page", "save_finding"}
    assert not [n for n in dir(Tools) if "send" in n or "delete" in n or "update" in n]


def test_llm_falls_through_to_the_next_model():
    class H:
        def __init__(self):
            self.n = 0

        def post(self, url, **kw):
            self.n += 1
            return Resp(b"{}", 429) if self.n == 1 else Resp(json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode())

    llm = LLM(env={"GROQ_API_KEY": "k", "GEMINI_API_KEY": "k"}, http=H(), sleep=lambda s: None)
    text, spec = llm.chat([{"role": "user", "content": "x"}])
    assert text == "hi" and spec.startswith("gemini:") and llm.log      # rate limit was waited out, then answered


def test_llm_with_no_keys_raises():
    try:
        LLM(env={}).chat([])
        assert False
    except LLMUnavailable:
        pass


def test_dry_run_store_writes_a_file(tmp_path):
    s = DryRunStore(out_dir=tmp_path)
    p = s.save_run({"findings": [], "steps": [], "stopped": "done", "model": None, "seconds": 0, "summary": ""})
    assert json.loads(open(p).read())["stopped"] == "done"


def test_cut_off_answer_is_retried_with_more_room():
    seen = []

    class H:
        def post(self, url, **kw):
            seen.append(kw["json"]["max_tokens"])
            first = len(seen) == 1
            return Resp(json.dumps({"choices": [{"finish_reason": "length" if first else "stop",
                                                 "message": {"content": "" if first else "ok"}}]}).encode())

    llm = LLM(env={"GROQ_API_KEY": "k"}, http=H(), sleep=lambda s: None)
    text, spec = llm.chat([{"role": "user", "content": "x"}], max_tokens=1000)
    assert text == "ok" and seen == [1000, 3000] and spec == "groq:openai/gpt-oss-120b"


def test_a_hard_failure_moves_to_the_next_model():
    class H:
        def __init__(self):
            self.n = 0

        def post(self, url, **kw):
            self.n += 1
            return Resp(b"{}", 401) if "generativelanguage" in url else Resp(json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode())

    llm = LLM(env={"GROQ_API_KEY": "k", "GEMINI_API_KEY": "k"}, http=H(), sleep=lambda s: None)
    text, spec = llm.chat([{"role": "user", "content": "x"}])
    assert spec.startswith("groq:") and any("HTTP 401" in l for l in llm.log)


def test_empty_stop_goes_to_next_model_without_killing_the_first():
    class H:
        def post(self, url, **kw):
            body = {"choices": [{"finish_reason": "stop", "message": {"content": "" if "gpt-oss" in kw["json"]["model"] else "hi"}}]}
            return Resp(json.dumps(body).encode())

    llm = LLM(env={"GROQ_API_KEY": "k"}, http=H(), sleep=lambda s: None)
    text, spec = llm.chat([{"role": "user", "content": "x"}])
    assert text == "hi" and spec == "groq:qwen/qwen3.8-27b"
    assert "groq:openai/gpt-oss-120b" not in llm.dead and any("empty answer" in l for l in llm.log)


def test_a_failed_call_is_not_repeated_and_the_error_is_recorded():
    http = FakeHTTP(Resp(b"", 404))
    t = tools(http=http)
    call = act("fetch_page", url="https://www.iaasb.org/nope")
    r = run(FakeLLM([call, call, {"done": True, "summary": "x"}]), t, "g", dict(LIMITS))
    assert len(http.calls) == 1                                           # second identical call never hit the site
    assert r["steps"][0]["ok"] is False and "404" in r["steps"][0]["error"] and r["steps"][1]["ok"] is False


def test_answer_written_into_the_reasoning_field_is_recovered():
    class H:
        def post(self, url, **kw):
            msg = {"content": "", "reasoning": 'We need to read first.{"tool":"read_tracker","args":{"days":7}}'}
            return Resp(json.dumps({"choices": [{"finish_reason": "stop", "message": msg}]}).encode())

    text, spec = LLM(env={"GROQ_API_KEY": "k"}, http=H(), sleep=lambda s: None).chat([{"role": "user", "content": "x"}])
    assert parse_action(text) == {"tool": "read_tracker", "args": {"days": 7}}


def test_parse_action_tolerates_think_blocks_and_chatter():
    assert parse_action('<think>maybe {"a": 1}</think>Sure! ```json\n{"tool": "search_news", "args": {"query": "x"}}\n```')["tool"] == "search_news"
    assert parse_action("no json here") is None


def test_save_finding_by_search_result_id():
    t = tools()
    res = t.call("search_news", {"query": "a"})
    assert "url" not in res["results"][0] and res["results"][0]["id"] == "r1"     # no long links in the model's context
    out = t.call("save_finding", {"ref": "r1", "summary": "s", "reason": "r", "theme": "ethics_standards"})
    assert out == {"saved": True, "total": 1} and t.findings[0]["url"]
    assert "error" in t.call("save_finding", {"ref": "r99", "summary": "s", "reason": "r"})


def test_agent_cannot_finish_before_minimum_searches():
    t = tools()
    r = run(FakeLLM([{"done": True, "summary": "nothing"}, act("search_news", query="a"), act("search_news", query="b"),
                     {"done": True, "summary": "ok"}]), t, "g", dict(LIMITS, min_searches=2))
    assert r["stopped"] == "done" and r["searches"] == 2 and r["steps"][0]["ok"] is False


def test_empty_handed_finish_gets_one_second_look():
    t = tools()
    llm = FakeLLM([act("search_news", query="a"), {"done": True, "summary": "nothing"},
                   act("save_finding", ref="r1", summary="s", reason="r"), {"done": True, "summary": "one"}])
    r = run(llm, t, "g", dict(LIMITS, min_searches=1))
    assert r["stopped"] == "done" and len(r["findings"]) == 1


def test_empty_handed_finish_is_accepted_the_second_time():
    t = tools()
    llm = FakeLLM([act("search_news", query="a"), {"done": True, "summary": "n"}, {"done": True, "summary": "still n"}])
    r = run(llm, t, "g", dict(LIMITS, min_searches=1))
    assert r["stopped"] == "done" and r["findings"] == [] and r["summary"] == "still n"


def test_arguments_written_beside_the_tool_name_are_accepted():
    t = tools()
    llm = FakeLLM([{"tool": "search_news", "query": "IAASB"}, {"tool": "search_news", "arguments": {"query": "IESBA"}},
                   {"done": True, "summary": "x"}])
    r = run(llm, t, "g", dict(LIMITS, min_searches=2))
    assert r["searches"] == 2 and r["steps"][0]["ok"] and r["steps"][1]["ok"]


def test_bad_arguments_error_shows_the_exact_shape():
    out = tools().call("search_news", {})
    assert "error" in out and '"args"' in out["error"]
