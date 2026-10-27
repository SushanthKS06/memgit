"""No-hardcode guard: tunables live in cfg / defaults.toml, never in source text.

Scans every memgit package source file for strings that, if they ever appeared
in logic, would mean a defaulted value escaped the configuration layer
(memory dir, markdown section headings, ids/prefixes, log paths, ports,
thresholds, model message templates...). Wiring code touches none of these —
they flow from config — so a hit means a regression.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import memgit

PACKAGE_ROOT = Path(memgit.__file__).parent

FORBIDDEN = [
    "./memory",              # memory.dir default
    "## Active",             # memory.active_heading
    "## Superseded",         # memory.superseded_heading
    "| superseded_by:",       # field_sep + memory.superseded_by_prefix render
    "[mem-",                 # memory.id_open + id_prefix
    "| s:",                  # memory.source_prefix line format
    '"- "',                  # memory.bullet
    "memgit agent",          # git.author_name
    "memgit@example.com",    # git.author_email
    "usage.jsonl",           # logging.usage_log_path
    "rejections.jsonl",      # logging.rejections_log_path
    "127.0.0.1",             # web.host
    "8377",                  # web.port
    "0.85",                  # verify.dup_similarity
    "/chat/completions",     # llm.chat_endpoint
    "Bearer {key}",          # llm.auth_header_template
    "consolidate.md",        # prompts.consolidate
    "context.md",            # prompts.recall_block
    "recall_select.md",      # prompts.recall_select
    "answer.md",             # prompts.answer
    "commit_message.md",     # prompts.commit_message
    "judge_recall.md",       # prompts.judge_recall
    "different context md fallback",
    "sounds good",           # verify.affirmations entry
    "let's do it",           # verify.affirmations entry
]


def source_files():
    return sorted(PACKAGE_ROOT.rglob("*.py"))


@pytest.mark.parametrize("needle", FORBIDDEN)
def test_source_has_no_hardcoded_tunables(needle: str):
    offenders = []
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        if needle in text:
            offenders.append(str(path.relative_to(PACKAGE_ROOT)))
    assert not offenders, f"{needle!r} hardcoded in {offenders}; it must come from config/defaults.toml"


def test_source_files_exercised():
    files = source_files()
    assert files, "memgit package sources not found"
    assert any(p.name == "store.py" for p in files)
    assert any(p.name == "llm.py" for p in files)
