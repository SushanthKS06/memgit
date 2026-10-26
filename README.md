# memgit — agent memory you can diff, audit, and roll back

Long-lived AI agents accumulate facts across many chats. Conversation history is
the wrong database for that: no dedup, no contradiction handling, no audit trail,
no undo.

**memgit** keeps agent memory in a plain **git repository of markdown files**.
Every chat session ends in exactly one memory commit — the model proposes facts
as JSON, memgit verifies each proposal against verbatim quotes from the
transcript, rejects hallucinations, and commits what survives. You can browse
history, diff any session, and **revert a bad session with one click**.

- Open models only: any OpenAI-compatible chat API (auto-detects `/models` discovery + `/chat/completions`)
- No config sprawl: every tunable lives in [memgit/defaults.toml](memgit/defaults.toml); override per project with `./memgit.toml`, `MEMGIT_SECTION__KEY` env vars, or `--set section.key=value`
- Fail loudly: invalid config/index/reply aborts the command immediately; no silent fallbacks
- Zero heavyweight deps: `pydantic`, `httpx`, `git`; stdlib servers for MCP + web

## Quickstart

```pwsh
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env           # then set MEMGIT_API_KEY="sk-..."

python -m memgit.cli doctor      # config, git, prompts, api, model discovery

python -m memgit.cli chat        # interactive; '/exit' ends the session and commits memory

python -m memgit.cli recall "which editor do I use?"   # preview the memory block
python -m memgit.cli log                          # memory commit history
python -m memgit.cli diff <sha>                   # what a session stored
python -m memgit.cli show <sha>                   # full message + patch
python -m memgit.cli revert <sha>                 # undo a bad session (creates a revert commit)
```

`memgit chat` is the primary loop: every turn recalls relevant memory, answers
with context available, collects the finished transcript, then consolidates:

```
transcript --> consolidate (model JSON proposal)
           --> verify   (verbatim quotes, paths, targets, duplicates)
           --> apply    (markdown files, atomic rewrite)
           --> commit   (one commit: +added ~superseded -rejected counts)
```

Rejected ops are not silently dropped: they're logged to `logs/rejections.jsonl`
and surfaced in the web UI.

## Reading through the agent

At recall time the Agent picks relevant files from a regenerated index
(`memory/index.md`), then only active lines get truncated by token budget
(`memory.max_recall_tokens`) so context stays fresh. The prompt file that wraps
them is `memgit/prompts/context.md`; you can point it at your own template with
`--set prompts.recall_block=my_context.md --prompts-dir ...`.

## MCP server

```pwsh
python -m memgit.cli serve
```

Stdin/stdout MCP server (JSON-RPC, protocol `2024-11-05`), tools:

| tool | purpose |
| ----------- | -------------------------------------------------- |
| `memory_recall` | fetch verbatim memory context for a query |
| `remember_session` | consolidate a finished transcript into memory |
| `memory_history` | list memory commits |
| `memory_find` | text-search commit messages |
| `memory_diff` | patch of a commit |
| `revert_commit` | roll back a bad session |

Register it in any MCP client with a stdio command like
`python -m memgit.cli serve` (cwd = your project).

## Web UI

```pwsh
python -m memgit.cli web          # binds [web] host/port (default 127.0.0.1:8377)
```

History with per-commit patch pages and one-click **revert** buttons, memory
overview, and the rejections audit page.

## Benchmarks (honesty policy: no cherry-picking, no hardcoding)

```pwsh
python -m bench.gen_data                        # synthetic session-transcript data (seeded)
python -m bench.viability                       # consolidation parsing/verification gate
python -m bench.gen_data --mode scenarios       # recovery scenarios (probes with expected facts)
python -m bench.report                          # runs scenarios x reps -> benchmark.csv/.md
```

Scores: `reverified` (fact physically present in committed memory), `verbalized`
(judge model rates the answer), `abstain`, `wrong`. Charts deliberately excluded.
Benchmarking runs through the exact same pipeline as production paths.

Heads up: `viability`, `bench.report`, and doctor's API probes need real
credentials — without them they exit 2 with a clear message instead of
pretending.

## The config picture

Default values (never duplicated in code — a guard test enforces this):

```toml
[llm]      base_url, api_key_env, model/judge_model, retries, temperatures, endpoints...
[prompts]  file names of every prompt/template
[memory]   dir layout, markdown separators, id prefixes, recall budgets...
[verify]   min_quote_chars, dup_similarity, normalise_rules
[recall]   fallback_grep, grep_min_token
[git]      author, timeout, autocrlf, log_limit
[web] / [mcp] / [logging] / [bench]
```

Layering: `--set` CLI overrides > `MEMGIT_SECTION__KEY` env vars > `./memgit.toml` > `defaults.toml`.
Secrets live in `.env` as env vars only (see `.env.example`); keys are never
printed or persisted inside config layers by doctor output.

## The memory format

A markdown file (`memory/*.md`) is:

```
# tools

## Active

- [mem-12] The user's main editor is VS Code | s:mem-sess3 | t:2 | "I use VS Code as my main editor for coding work"

## Superseded

- [mem-4] The user prefers nano | s:mem-sess1 | t:1 | "I use nano daily" | superseded_by:mem-12
```

Every line carries its source session, transcript turn and verbatim quote;
superseded lines keep provenance. `index.md` is regenerated from file headers,
never hand-edited. All markdown keys/rules are configurable (`[memory]`), the
format is round-trip tested.

## Development

```pwsh
python -m pytest -q              # full suite (offline: LLM traffic via MockTransport)
scripts\demo.ps1                 # offline end-to-end demo of sessions/log/diff/revert
```

Layout:

```
memgit/
  core/    schema, consolidate, verify, store(git), session, retrieve, answer
  mcp/     stdio MCP server (tools slice)
  web/     stdlib HTTP UI
  prompts/ every prompt + few-shots + fix-retry + commit-message templates
bench/     gen_data, viability gate, recovery benchmark
```

## Honesty receipts

- Entire suite runs offline (mocked transports); real-LLM parity requires credentials — nothing faked.
- `memgit.toml.example` shows the endpoint/key flow for custom providers.
- Git history equals product history: every design phase landed as one commit.
