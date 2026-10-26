# Offline end-to-end demo of memgit: seeds a throwaway memory repo with two
# sessions, then walks through the history tools (log / diff / show / revert).
# No API key needed — the pipeline under test is the git store, not the model.
# Real-LLM flows need credentials (see README).
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$demoDir = Join-Path ([IO.Path]::GetTempPath()) "opencode\memgit-demo"

if (Test-Path $demoDir) { Remove-Item -Recurse -Force $demoDir }
New-Item -ItemType Directory -Force -Path $demoDir | Out-Null
Write-Host "== demo workspace: $demoDir"

# 1) seed two sessions (deterministic content, no LLM)
$seed = @'
from pathlib import Path

from memgit.cli import bootstrap_settings
from memgit.core.store import MemoryStore

cfg = bootstrap_settings(None, None)
store = MemoryStore(cfg.settings)
mem = Path(cfg.settings.memory.dir)
mem.mkdir(parents=True, exist_ok=True)

store.begin_session()
(mem / "notes.md").write_text(
    '# notes\n\n## Active\n\n'
    '- [mem-1] The user codes with Neovim daily | s:mem-sess1 | t:2 | "Neovim is all I code with"\n\n'
    '## Superseded\n\n',
    encoding="utf-8",
)
store.commit_all("memory session mem-sess1\n\n+1 ~0 -0 (noop 0)\nlearned [mem-1] notes.md")
store.finish_session()

store.begin_session()
(mem / "notes.md").write_text(
    '# notes\n\n## Active\n\n'
    '- [mem-2] The user codes with Neovim daily and evenings | s:mem-sess2 | t:1 | "Neovim is all I code with"\n\n'
    '## Superseded\n\n\n- [mem-1] The user codes with Neovim daily | s:mem-sess1 | t:2 | "Neovim is all I code with" | superseded_by:mem-2\n',
    encoding="utf-8",
)
store.commit_all("memory session mem-sess2\n\n+1 ~1 -0 (noop 0)\nlearned [mem-2] notes.md\nwas  [mem-1]")
store.finish_session()
print("seeded 2 sessions")
'@
Set-Content -Path (Join-Path $demoDir "seed.py") -Value $seed -Encoding UTF8
Push-Location $repoRoot
try {
    Push-Location $demoDir
    python seed.py
    if ($LASTEXITCODE -ne 0) { throw "seeding failed" }

    # 2) walk the tools that do not need the model
    Write-Host "`n== memgit log =="
    python -m memgit.cli log
    if ($LASTEXITCODE -ne 0) { throw "log failed" }

    $first = (git -C (Join-Path $demoDir "memory") log --pretty=%H --reverse | Select-Object -First 1)
    Write-Host "`n== memgit diff $first =="
    python -m memgit.cli diff $first
    if ($LASTEXITCODE -ne 0) { throw "diff failed" }

    $newest = (git -C (Join-Path $demoDir "memory") log --pretty=%H | Select-Object -First 1)
    Write-Host "`n== memgit revert $newest =="
    python -m memgit.cli revert $newest
    if ($LASTEXITCODE -ne 0) { throw "revert failed" }

    Write-Host "`n== memgit log after revert =="
    python -m memgit.cli log
    if ($LASTEXITCODE -ne 0) { throw "log failed" }

    Write-Host "`n== try UI: run 'python -m memgit.cli --set web.port=8377 web' in $demoDir =="
}
finally {
    Pop-Location
    Pop-Location
}
Write-Host "`nDemo finished. Scratch store left at $demoDir\memory for inspection."
