"""Regression tests for git argument injection via resolve_commit.

Covers: flag injection, shell metacharacters, HEAD/branch names, empty string,
valid short SHA, valid full SHA, nonexistent SHA.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from memgit.core.store import MemoryStore, StoreError


@pytest.fixture()
def store(settings_factory, tmp_path: Path) -> MemoryStore:
    s = settings_factory(f"memory.dir={tmp_path / 'mem'}")
    return MemoryStore(s, path=tmp_path / "mem")


def make_commit(store: MemoryStore) -> str:
    """Create a real git commit and return its full SHA."""
    path = store.root / "notes.md"
    path.write_text(
        '# notes\n\n## Active\n\n- [mem-1] Test fact | s:mem-sess1 | t:1 | "test quote here"\n\n## Superseded\n',
        encoding="utf-8",
    )
    store.begin_session()
    sha = store.commit_all("test commit for injection tests")
    store.finish_session()
    return sha


# ---------------------------------------------------------------------------
# Injection / malformed refs – all must raise StoreError BEFORE touching git
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_ref",
    [
        "",                         # empty
        "   ",                      # whitespace only
        "--output=/tmp/x",          # flag injection
        "-m",                       # short flag
        "HEAD;ls",                  # shell injection with semicolon
        "../x",                     # path traversal
        "HEAD",                     # symbolic ref (not hex)
        "main",                     # branch name
        "refs/heads/main",          # full ref
        "abc123!@#",                # special characters
        "aaaaaaa" * 6,              # 42 chars: too long
        "aaaaaa",                   # 6 chars: too short
    ],
)
def test_resolve_commit_rejects_bad_ref(store: MemoryStore, bad_ref: str) -> None:
    """Malformed / injection refs must be rejected with StoreError before any git call."""
    make_commit(store)  # ensure the repo exists
    with pytest.raises(StoreError, match=r"(unsafe|invalid|non-empty)"):
        store.resolve_commit(bad_ref)


# ---------------------------------------------------------------------------
# Valid refs
# ---------------------------------------------------------------------------


def test_resolve_commit_accepts_full_sha(store: MemoryStore) -> None:
    """A valid 40-char hex SHA resolves to itself."""
    full = make_commit(store)
    assert len(full) == 40
    resolved = store.resolve_commit(full)
    assert resolved == full


def test_resolve_commit_accepts_short_sha(store: MemoryStore) -> None:
    """A valid 7-char short SHA resolves to the full 40-char SHA."""
    full = make_commit(store)
    short = full[:7]
    resolved = store.resolve_commit(short)
    assert resolved == full


# ---------------------------------------------------------------------------
# Nonexistent SHA
# ---------------------------------------------------------------------------


def test_resolve_commit_rejects_nonexistent_sha(store: MemoryStore) -> None:
    """A syntactically valid hex ref that doesn't exist in the repo raises StoreError."""
    make_commit(store)
    # 40 hex chars, but not a real commit
    fake = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    with pytest.raises(StoreError):
        store.resolve_commit(fake)


# ---------------------------------------------------------------------------
# Transitive: show / diff_against / revert all use resolve_commit
# ---------------------------------------------------------------------------


def test_show_rejects_injection(store: MemoryStore) -> None:
    make_commit(store)
    with pytest.raises(StoreError):
        store.show("--output=/tmp/evil")


def test_diff_against_rejects_injection(store: MemoryStore) -> None:
    make_commit(store)
    with pytest.raises(StoreError):
        store.diff_against("HEAD;id")


def test_revert_rejects_injection(store: MemoryStore) -> None:
    make_commit(store)
    with pytest.raises(StoreError):
        store.revert("HEAD")
