import subprocess
from pathlib import Path

import pytest

from memgit.cli import main, bootstrap_settings
from memgit.core.store import MemoryStore


NOTE_TEXT_A = (
    "# notes\n\n## Active\n\n"
    '- [mem-1] The user keeps notes in git | s:mem-sess1 | t:1 | "notes in git"\n\n'
    "## Superseded\n\n"
)
NOTE_TEXT_B = (
    "# notes\n\n## Active\n\n"
    '- [mem-2] The user keeps notes in git and journal apps | s:mem-sess2 | t:1 | "notes and journals"\n\n'
    "## Superseded\n\n\n- [mem-1] The user keeps notes in git | s:mem-sess1 | t:1 | \"notes in git\" | superseded_by:mem-2\n"
)


def seed_two_sessions(root: Path) -> MemoryStore:
    settings = bootstrap_settings(None, None).settings
    store = MemoryStore(settings, path=root)
    store.begin_session()
    (root / "notes.md").write_text(NOTE_TEXT_A, encoding="utf-8")
    store.commit_all("memory session mem-sess1\n\n+1 ~0 -0 (noop 0)\nlearned [mem-1] notes.md")
    store.finish_session()
    store.begin_session()
    (root / "notes.md").write_text(NOTE_TEXT_B, encoding="utf-8")
    store.commit_all("memory session mem-sess2\n\n+1 ~1 -0 (noop 0)\nlearned [mem-2] notes.md\nwas  [mem-1]")
    store.finish_session()
    return store


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield


def shas(root: Path) -> list[str]:
    log = subprocess.run(
        ["git", "log", "--pretty=%H%x20%s"], cwd=root, capture_output=True, text=True, encoding="utf-8", check=False
    )
    return [line for line in log.stdout.splitlines() if line.strip()]


@pytest.fixture()
def store(tmp_path: Path) -> MemoryStore:
    return seed_two_sessions(tmp_path / "memory")


def test_log_lists_commits_newest_first(store: MemoryStore, tmp_path: Path, capsys):
    rc = main(["log"])
    out = capsys.readouterr().out
    assert rc == 0
    listed = [line for line in out.splitlines() if line.strip()]
    assert len(listed) == 2
    assert "memory session mem-sess2" in listed[0]
    assert "memory session mem-sess1" in listed[1]


def test_log_limit(store: MemoryStore, tmp_path: Path, capsys):
    rc = main(["log", "--limit", "1"])
    out = capsys.readouterr().out
    listed = [line for line in out.splitlines() if line.strip()]
    assert rc == 0
    assert len(listed) == 1
    assert "mem-sess2" in listed[0]


def test_log_empty(tmp_path: Path, capsys):
    (tmp_path / "memory").mkdir()
    rc = main(["log"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "(no commits yet)" in out


def test_log_bad_limit(store: MemoryStore, capsys):
    rc = main(["log", "--limit", "0"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "--limit must be >= 1" in err


def test_diff_shows_added_lines(store: MemoryStore, capsys):
    first = shas(store.root)[-1].split()[0]
    rc = main(["diff", first])
    out = capsys.readouterr().out
    assert rc == 0
    assert "notes.md" in out
    assert '"notes in git"' in out
    assert "+## Active" in out


def test_diff_commit_without_changes(store: MemoryStore, capsys):
    store.begin_session()
    empty_sha = store.commit_all("memory session mem-sess0 (empty)")
    store.finish_session()
    rc = main(["diff", empty_sha])
    out = capsys.readouterr().out
    assert rc == 0
    assert "(no changes vs parent)" in out


def test_show_prints_message_and_patch(store: MemoryStore, capsys):
    first = shas(store.root)[-1].split()[0]
    rc = main(["show", first])
    out = capsys.readouterr().out
    assert rc == 0
    assert "memory session mem-sess1" in out
    assert "diff --git" in out


def test_revert_last_commit(store: MemoryStore, capsys):
    last = shas(store.root)[0].split()[0]
    rc = main(["revert", last])
    out = capsys.readouterr().out
    assert rc == 0
    assert "revert committed:" in out
    assert '[mem-2] The user keeps notes in git and journal apps' not in (store.root / "notes.md").read_text(encoding="utf-8")
    commits = shas(store.root)
    assert len(commits) == 3
    assert "Revert" in commits[0]


def test_revert_conflicting_commit_fails_clean(store: MemoryStore, capsys):
    first = shas(store.root)[-1].split()[0]
    rc = main(["revert", first])
    err = capsys.readouterr().err
    assert rc == 1
    assert "revert failed:" in err
    assert len(shas(store.root)) == 2
    proc = subprocess.run(
        ["git", "status", "--porcelain"], cwd=store.root, capture_output=True, text=True, encoding="utf-8", check=False
    )
    assert proc.stdout.strip() == ""


def test_config_error_guards_non_doctor_commands(tmp_path: Path, capsys):
    (tmp_path / "memgit.toml").write_text('memory = 1337\n', encoding="utf-8")
    rc = main(["log"])
    err = capsys.readouterr().err
    assert rc == 2
    assert err
