-- CAPA Intelligence Agent: two new tables. Run once in the SAME Supabase project as the PAO News Monitor
-- (SQL Editor > New query > paste > Run). Safe to run more than once. Touches nothing that already exists.
--
-- Staff can READ these (a future "Agent suggestions" view). Only the agent's secret key can WRITE them,
-- because no insert/update policy exists for signed-in users.

create table if not exists agent_runs (
  id         bigserial primary key,
  created_at timestamptz not null default now(),
  stopped    text,          -- done / max_steps / time_limit / ai_unavailable / bad_output
  model      text,
  seconds    numeric,
  summary    text,
  steps      jsonb          -- what the agent did, step by step
);

create table if not exists agent_findings (
  id          bigserial primary key,
  created_at  timestamptz not null default now(),
  run_id      bigint references agent_runs(id) on delete cascade,
  title       text not null,
  url         text not null,
  summary     text,
  reason      text,
  theme       text,
  status      text not null default 'suggested' check (status in ('suggested','approved','rejected')),
  reviewed_by text,
  reviewed_at timestamptz
);
create index if not exists agent_findings_status on agent_findings (status, created_at desc);

alter table agent_runs     enable row level security;
alter table agent_findings enable row level security;

drop policy if exists "team reads agent runs" on agent_runs;
create policy "team reads agent runs" on agent_runs for select to authenticated using (true);
drop policy if exists "team reads agent findings" on agent_findings;
create policy "team reads agent findings" on agent_findings for select to authenticated using (true);

notify pgrst, 'reload schema';
