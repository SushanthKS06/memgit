"""Filesystem utilities: cross-version, cross-platform helpers.

All tunables/constants must live in config; nothing here is hardcoded beyond
the shutil/stat imports and Python stdlib calls.
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path


def rmtree_force(path: Path) -> None:
    """Remove a directory tree, clearing read-only bits before retry.

    Works on Python 3.11 (uses onerror=) and Python 3.12+ (uses onexc=).
    Git marks objects read-only on Windows; this handles that transparently.
    """
    if not path.exists():
        return

    def _clear_readonly_and_retry_onerror(func, failing_path, exc_info) -> None:
        """Python 3.11 onerror= callback signature: (func, path, excinfo)."""
        try:
            os.chmod(failing_path, stat.S_IWRITE)
            func(failing_path)
        except OSError:
            pass

    def _clear_readonly_and_retry_onexc(func, failing_path, exc) -> None:
        """Python 3.12+ onexc= callback signature: (func, path, exc)."""
        try:
            os.chmod(failing_path, stat.S_IWRITE)
            func(failing_path)
        except OSError:
            pass

    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_clear_readonly_and_retry_onexc)
    else:
        shutil.rmtree(path, onerror=_clear_readonly_and_retry_onerror)
