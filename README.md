# CAPA Intelligence Agent

A small research agent that sits **next to** the PAO News Monitor (separate repo, same Supabase project). It is an
experiment: it does not change, replace or send anything on the tracker.

## What it does
Once per run it is given a goal (see `agent/config.py`): find developments on AI and digitalisation, talent,
sustainability and ethics/standards that CAPA should know about, and that the tracker does not already hold.
Each turn the AI model picks **one tool**; the code runs it and shows the result; repeat until it says it is done
or a limit is reached.

| Tool | What it does | Writes? |
|---|---|---|
| `read_tracker` | Reads items the tracker already collected (last N days) | no |
| `search_news` | Google News search (last 30 days) | no |
| `fetch_page` | Text and links of one public page, **allowlisted sites only** (IAASB, IESBA, IFAC, IFRS) | no |
| `save_finding` | Saves one suggestion, status `suggested`, for staff to review | its own table only |

There is no tool to send email, edit tracker items or touch people's records.

## Safety rules (enforced in code, tested in `tests/test_agent.py`)
- Limits per run: 12 steps, 3 searches, 4 page fetches, 10 findings, 480 seconds.
- A finding's URL must have appeared in a tool result: it cannot cite a source it did not find.
- Pages outside the allowlist are refused. Change it with the `AGENT_ALLOWED_DOMAINS` variable.
- If every AI model runs out of free quota, the run stops. It never switches to a paid service.
- Manual trigger only (GitHub Actions > Run intelligence agent). Nothing runs on a schedule yet.

## AI models (free tiers)
Fallback order, used only if the key exists: Google Gemini Flash (most generous free limits) > Groq (`gpt-oss-120b`, then Qwen) >
Cloudflare Workers AI (Llama 3.3 70B) > OpenRouter free (NVIDIA Nemotron). Override with `AGENT_MODELS`.

## Try it
```
pip install -r requirements.txt
export GROQ_API_KEY=...            # any one provider key is enough
python -m agent --dry-run          # no database; result written to out/run-*.json
python -m pytest                   # 23 tests, no keys or network needed
```

## Saving to Supabase
1. Run `supabase/agent-tables.sql` once in the tracker's Supabase project (adds `agent_runs` and `agent_findings`).
2. Add repository secrets: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` and at least one AI key
   (`GROQ_API_KEY`, `GEMINI_API_KEY`, `CF_ACCOUNT_ID` + `CF_AI_TOKEN`, `OPENROUTER_API_KEY`).
3. Actions > Run intelligence agent > untick **Dry run**.

Known limitation: v1 uses the Supabase **service key**, which can do anything in that project. The only code path that
writes is `agent/store.py`. Before wider use, replace it with a database role limited to these two tables.

## Not built yet
A view in the tracker for approving or rejecting suggestions; a schedule; the member engagement agent.

## Planning
BMAD is installed (`_bmad/`). Planning documents go in `_bmad-output/`.
