"""Memgit CLI. Phase 0: doctor. Further subcommands arrive with later phases."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from memgit.config import ConfigError, LoadedConfig, load_env_file, load_settings
from memgit.core.llm import LLMClient, LLMError
from memgit.core.schema import ensure_layout


@dataclass
class CheckRow:
    name: str
    status: str  # ok | fail | warn
    detail: str


def bootstrap_settings(config_file: Optional[str], cli_overrides: Optional[List[str]]) -> LoadedConfig:
    """Two-pass load: resolve [logging].env_file from layer 1-3, load it, then fold env layer."""
    first = load_settings(config_file, cli_overrides)
    load_env_file(Path(first.settings.logging.env_file))
    return load_settings(config_file, cli_overrides)


def check_config(cfg: Optional[LoadedConfig]) -> CheckRow:
    if cfg is None:
        return CheckRow("config", "fail", "configuration failed to load")
    return CheckRow("config", "ok", "layers: " + ", ".join(cfg.layers))


def check_prompts(cfg: LoadedConfig) -> List[CheckRow]:
    from memgit.config import prompts_dir

    root = prompts_dir(cfg.settings)
    rows: List[CheckRow] = []
    missing: List[str] = []
    for field_name in type(cfg.settings.prompts).model_fields:
        filename = getattr(cfg.settings.prompts, field_name)
        if not (root / filename).is_file():
            missing.append(f"{field_name}={filename}")
    if missing:
        rows.append(
            CheckRow(
                "prompts",
                "fail",
                f"missing in {root}: " + ", ".join(missing) + " (set [memory].prompts_dir to override)",
            )
        )
    else:
        rows.append(CheckRow("prompts", "ok", f"{len(type(cfg.settings.prompts).model_fields)} files under {root}"))
    return rows


def check_memory_dir(cfg: LoadedConfig) -> CheckRow:
    path = Path(cfg.settings.memory.dir)
    try:
        ensure_layout(path, cfg.settings)
        probe = path / cfg.settings.memory.state_dir / ".doctor-probe"
        probe.write_text("probe", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return CheckRow("memory dir", "fail", f"{path}: {exc}")
    return CheckRow("memory dir", "ok", str(path.resolve()))


def check_git() -> CheckRow:
    binary = shutil.which("git")
    git_exe = binary if binary is not None else ""
    if not git_exe:
        return CheckRow("git", "fail", "git not found on PATH; install git or add it to PATH")
    try:
        proc = subprocess.run(
            [git_exe, "--version"], capture_output=True, text=True, timeout=10, check=True
        )
        dunder = strip_carriage(proc.stdout.strip())
        if not dunder.startswith("git "):
            raise ValueError(f"unexpected git --version output: {dunder}")
        return CheckRow("git", "ok", dunder)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return CheckRow("git", "fail", f"git found but unusable: {exc}")


def strip_carriage(text: str) -> str:
    return text.replace("\r", "")


def check_api_key(cfg: LoadedConfig) -> CheckRow:
    name = cfg.settings.llm.api_key_env
    value = os.environ.get(name, "")
    if not value.strip():
        return CheckRow(
            "api key",
            "fail",
            f"env var {name!r} is empty; copy .env.example to .env, set {name}=..., and rerun",
        )
    return CheckRow("api key", "ok", f"env var {name!r} present (value not shown)")


def check_discovery(cfg: LoadedConfig, client: LLMClient) -> List[CheckRow]:
    rows: List[CheckRow] = []
    base = cfg.settings.llm.base_url.strip()
    if not base:
        rows.append(CheckRow("api endpoint", "fail", "llm.base_url is empty; set [llm].base_url in memgit.toml"))
        return rows
    try:
        models = client.discover_models()
    except LLMError as exc:
        rows.append(CheckRow("api endpoint", "fail", str(exc)))
        rows.append(CheckRow("models", "fail", "skipped: discovery failed"))
        rows.append(CheckRow("model selection", "fail", "skipped: discovery failed"))
        return rows
    endpoint = base + cfg.settings.llm.models_endpoint
    rows.append(CheckRow("api endpoint", "ok", f"{endpoint} reachable"))
    rows.append(
        CheckRow("models", "ok", f"{len(models)} model(s): {', '.join(models[:8])}")
        if models
        else CheckRow("models", "fail", "no models returned")
    )
    try:
        consolidation = client.resolve_model("consolidation")
        judge = client.resolve_model("judge")
    except LLMError as exc:
        rows.append(CheckRow("model selection", "fail", str(exc)))
        return rows
    how = "configured" if cfg.settings.llm.model.strip() and cfg.settings.llm.judge_model.strip() else "auto-selected"
    rows.append(
        CheckRow("model selection", "ok", f"{how}: consolidation={consolidation} judge={judge}")
    )
    return rows


def print_table(rows: List[CheckRow]) -> None:
    name_w = max(len(r.name) for r in rows)
    status_w = max(len(r.status) for r in rows)
    detail_w = max(len(r.detail) for r in rows)
    print(f"{'CHECK':<{name_w}}  {'STATUS':<{status_w}}  DETAIL")
    print(f"{'-' * name_w}  {'-' * status_w}  {'-' * detail_w}")
    for row in rows:
        print(f"{row.name:<{name_w}}  {row.status:<{status_w}}  {row.detail}")


def run_doctor(cfg: Optional[LoadedConfig]) -> int:
    rows: List[CheckRow] = []
    config_row = check_config(cfg)
    rows.append(config_row)
    if cfg is None:
        print_table(rows)
        return 1
    rows.extend(check_prompts(cfg))
    rows.append(check_memory_dir(cfg))
    rows.append(check_git())
    key_row = check_api_key(cfg)
    rows.append(key_row)
    if key_row.status != "ok":
        rows.append(CheckRow("api endpoint", "fail", f"skipped: {cfg.settings.llm.api_key_env!r} missing"))
        rows.append(CheckRow("models", "fail", "skipped: api key missing"))
        rows.append(CheckRow("model selection", "fail", "skipped: api key missing"))
    else:
        try:
            client = LLMClient(cfg.settings)
        except LLMError as exc:
            rows.append(CheckRow("api endpoint", "fail", str(exc)))
            rows.append(CheckRow("models", "fail", "skipped: client init failed"))
            rows.append(CheckRow("model selection", "fail", "skipped: client init failed"))
        else:
            rows.extend(check_discovery(cfg, client))
    has_fail = any(row.status == "fail" for row in rows)
    print_table(rows)
    return 1 if has_fail else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memgit", description="Agent memory you can diff, audit, and roll back")
    parser.add_argument("--config-file", default=None, help="explicit TOML config path (overrides memgit.toml lookup)")
    parser.add_argument("--set", dest="overrides", action="append", default=None, metavar="SECTION.KEY=VALUE", help="config override; repeatable")
    parser.add_argument("--prompts-dir", default=None, help="override the prompts dir (same as --set memory.prompts_dir=...)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="validate config, git, prompts, api endpoint and model discovery")
    return parser


def apply_prompts_dir_flag(argv: List[str]) -> List[str]:
    argv = list(argv)
    for idx, flag in enumerate(argv):
        if flag == "--prompts-dir":
            value = argv[idx + 1]
            argv[idx : idx + 2] = []
            argv.extend([f"--set", f"memory.prompts_dir={value}"])
            break
    return argv


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = apply_prompts_dir_flag(argv)
    args = build_parser().parse_args(argv)
    try:
        cfg = bootstrap_settings(args.config_file, args.overrides)
    except ConfigError as exc:
        cfg = None
        config_error = str(exc)
        print(config_error, file=sys.stderr)
    command = args.command
    if command == "doctor":
        return run_doctor(cfg)
    print(f"unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
