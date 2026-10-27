"""Regression tests for memgit/core/fsutil.py: rmtree_force works on read-only files."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from memgit.core.fsutil import rmtree_force


def test_rmtree_force_removes_normal_directory(tmp_path: Path) -> None:
    """rmtree_force removes a plain directory tree."""
    d = tmp_path / "toremove"
    d.mkdir()
    (d / "file.txt").write_text("hello", encoding="utf-8")
    (d / "sub").mkdir()
    (d / "sub" / "nested.txt").write_text("world", encoding="utf-8")
    rmtree_force(d)
    assert not d.exists()


def test_rmtree_force_removes_readonly_file(tmp_path: Path) -> None:
    """rmtree_force removes a tree containing read-only files (mirrors what git does)."""
    d = tmp_path / "toremove"
    d.mkdir()
    ro_file = d / "readonly.txt"
    ro_file.write_text("locked", encoding="utf-8")
    # Make the file read-only
    os.chmod(str(ro_file), stat.S_IREAD)
    rmtree_force(d)
    assert not d.exists()


def test_rmtree_force_noop_on_missing(tmp_path: Path) -> None:
    """rmtree_force does nothing if path does not exist (idempotent)."""
    missing = tmp_path / "does_not_exist"
    assert not missing.exists()
    rmtree_force(missing)  # must not raise
    assert not missing.exists()


def test_rmtree_force_removes_deeply_nested_readonly(tmp_path: Path) -> None:
    """rmtree_force works on nested read-only directories."""
    d = tmp_path / "nested"
    deep = d / "a" / "b" / "c"
    deep.mkdir(parents=True)
    ro = deep / "file.txt"
    ro.write_text("data", encoding="utf-8")
    os.chmod(str(ro), stat.S_IREAD)
    rmtree_force(d)
    assert not d.exists()
