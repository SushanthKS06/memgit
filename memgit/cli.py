"""Memgit CLI: doctor, chat, recall, history (log/diff/show/revert), MCP serve, web UI."""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

from memgit.config import ConfigError, LoadedConfig, load_env_file, load_settings
from memgit.core.consolidate import ROLE_USER
from memgit.core.llm import BudgetCounter, LLMClient, LLMError
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


class DoctorSanityJson(BaseModel):
    status: str


def check_chat_sanity(client: LLMClient) -> CheckRow:
    try:
        reply = client.chat([{"role": ROLE_USER, "content": "respond with 'pong'"}], purpose="doctor_ping")
        return CheckRow("chat sanity", "ok", f"replied in {reply.latency_s:.2f}s ({reply.model})")
    except Exception as exc:
        return CheckRow("chat sanity", "fail", str(exc))


def check_json_sanity(client: LLMClient) -> CheckRow:
    try:
        res, used_retry = client.chat_json(
            [{"role": ROLE_USER, "content": 'Output strict JSON only: {"status": "ok"}'}],
            schema=DoctorSanityJson,
            purpose="doctor_json",
        )
        retry_note = " (used retry)" if used_retry else ""
        return CheckRow("json sanity", "ok", f"valid (status={res.status!r}{retry_note})")
    except Exception as exc:
        return CheckRow("json sanity", "fail", str(exc))


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
        rows.append(CheckRow("chat sanity", "fail", "skipped: api key missing"))
        rows.append(CheckRow("json sanity", "fail", "skipped: api key missing"))
    else:
        try:
            client = LLMClient(cfg.settings)
        except LLMError as exc:
            rows.append(CheckRow("api endpoint", "fail", str(exc)))
            rows.append(CheckRow("models", "fail", "skipped: client init failed"))
            rows.append(CheckRow("model selection", "fail", "skipped: client init failed"))
            rows.append(CheckRow("chat sanity", "fail", "skipped: client init failed"))
            rows.append(CheckRow("json sanity", "fail", "skipped: client init failed"))
        else:
            disc_rows = check_discovery(cfg, client)
            rows.extend(disc_rows)
            if any(r.status == "fail" for r in disc_rows):
                rows.append(CheckRow("chat sanity", "fail", "skipped: discovery or model selection failed"))
                rows.append(CheckRow("json sanity", "fail", "skipped: discovery or model selection failed"))
            else:
                rows.append(check_chat_sanity(client))
                rows.append(check_json_sanity(client))
    has_fail = any(row.status == "fail" for row in rows)
    print_table(rows)
    return 1 if has_fail else 0


def cmd_chat(cfg: LoadedConfig, args: argparse.Namespace) -> int:
    from memgit.core.answer import Answerer
    from memgit.core.consolidate import Turn
    from memgit.core.retrieve import Recaller
    from memgit.core.session import MemorySession
    from memgit.core.store import MemoryStore

    settings = cfg.settings
    store = MemoryStore(settings)
    store.ensure_git()
    client = LLMClient(settings, budget=BudgetCounter())
    recaller = Recaller(settings, client)
    answerer = Answerer(settings, client)
    session = MemorySession(settings, store, client)
    transcript: List[Turn] = []
    print("memgit chat — type '/exit' on its own line to end the session and commit memory")
    while True:
        try:
            raw = input()
        except EOFError:
            break
        line = raw.rstrip("\n")
        if line.strip() == "/exit":
            break
        if not line.strip():
            continue
        user_turn = Turn(turn=len(transcript) + 1, role="user", text=line)
        candidate = transcript + [user_turn]
        try:
            memory_block = recaller.recall(line).block
        except LLMError as exc:
            print(f"[recall unavailable: {exc}]", file=sys.stderr)
            memory_block = ""
        reply = ""
        stream_chunks: List[str] = []
        try:
            for chunk in answerer.answer_stream(candidate, memory_block):
                sys.stdout.write(chunk)
                sys.stdout.flush()
                stream_chunks.append(chunk)
            sys.stdout.write("\n")
            sys.stdout.flush()
            reply = "".join(stream_chunks)
        except (LLMError, Exception):
            try:
                reply = answerer.answer(candidate, memory_block)
                print(reply)
            except LLMError as exc:
                print(f"[answer failed: {exc}]", file=sys.stderr)
                continue
        transcript = candidate + [Turn(turn=len(candidate) + 1, role="assistant", text=reply)]
    if not transcript:
        print("no turns; nothing to consolidate")
        return 0
    outcome = session.run(transcript)
    if outcome.ok:
        print(outcome.message)
        print(f"committed: {outcome.commit_sha}")
        return 0
    print(outcome.message, file=sys.stderr)
    return 1


def cmd_recall(cfg: LoadedConfig, args: argparse.Namespace) -> int:
    from memgit.core.llm import LLMClient as _Client
    from memgit.core.retrieve import Recaller
    from memgit.core.store import MemoryStore

    settings = cfg.settings
    store = MemoryStore(settings)
    store.ensure_git()
    client = LLMClient(settings, budget=BudgetCounter())
    recaller = Recaller(settings, client)
    try:
        result = recaller.recall(args.query)
    except LLMError as exc:
        print(f"recall failed: {exc}", file=sys.stderr)
        return 1
    print(result.block)
    print(f"files: {', '.join(result.files) if result.files else '(none)'}")
    print(f"tokens: {result.tokens} (used_llm={result.used_llm} used_grep={result.used_grep})")
    return 0


def cmd_serve(cfg: LoadedConfig, args: argparse.Namespace) -> int:
    from memgit.mcp.server import ServerCore, run_stdio

    core = ServerCore(cfg.settings)
    run_stdio(core)
    return 0


def cmd_web(cfg: LoadedConfig, args: argparse.Namespace) -> int:
    from memgit.web.server import serve

    print(f"memgit web UI on http://{cfg.settings.web.host}:{cfg.settings.web.port} (ctrl-c to stop)")
    logging.getLogger("memgit.web").setLevel(logging.DEBUG)
    serve(cfg.settings)
    return 0


def cmd_log(cfg: LoadedConfig, args: argparse.Namespace) -> int:
    from memgit.core.store import MemoryStore

    settings = cfg.settings
    limit = args.limit if args.limit is not None else settings.git.log_limit
    if limit <= 0:
        print("--limit must be >= 1", file=sys.stderr)
        return 2
    commits = MemoryStore(settings).log(limit)
    if not commits:
        print("(no commits yet)")
        return 0
    for commit in commits:
        print(f"{commit.short_sha}  {commit.date}  {commit.subject}")
    return 0


def cmd_diff(cfg: LoadedConfig, args: argparse.Namespace) -> int:
    from memgit.core.store import MemoryStore

    out = MemoryStore(cfg.settings).diff_against(args.commit)
    if not out.strip():
        print("(no changes vs parent)")
        return 0
    print(out)
    return 0


def cmd_show(cfg: LoadedConfig, args: argparse.Namespace) -> int:
    from memgit.core.store import MemoryStore

    print(MemoryStore(cfg.settings).show(args.commit))
    return 0


def cmd_revert(cfg: LoadedConfig, args: argparse.Namespace) -> int:
    from memgit.core.store import MemoryStore, StoreError

    store = MemoryStore(cfg.settings)
    try:
        new_sha = store.revert(args.commit)
    except StoreError as exc:
        print(f"revert failed: {exc}", file=sys.stderr)
        return 1
    print(f"revert committed: {new_sha}")
    return 0


def cmd_compact(cfg: LoadedConfig, args: argparse.Namespace) -> int:
    from memgit.core.store import MemoryStore, StoreError

    store = MemoryStore(cfg.settings)
    try:
        removed, sha = store.compact()
    except StoreError as exc:
        print(f"compact failed: {exc}", file=sys.stderr)
        return 1
    if removed == 0:
        print("no superseded memories to compact")
        return 0
    print(f"compacted {removed} superseded entries; commit {sha}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memgit", description="Agent memory you can diff, audit, and roll back")
    parser.add_argument("--config-file", default=None, help="explicit TOML config path (overrides memgit.toml lookup)")
    parser.add_argument("--set", dest="overrides", action="append", default=None, metavar="SECTION.KEY=VALUE", help="config override; repeatable")
    parser.add_argument("--prompts-dir", default=None, help="override the prompts dir (same as --set memory.prompts_dir=...)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="validate config, git, prompts, api endpoint and model discovery")
    sub.add_parser("chat", help="chat with the agent; ends with a memory consolidation commit")
    recall_cmd = sub.add_parser("recall", help="preview the memory context block for a query")
    recall_cmd.add_argument("query", help="query text; quoted if it contains spaces")
    log_cmd = sub.add_parser("log", help="list memory commits, newest first")
    log_cmd.add_argument("--limit", type=int, default=None, metavar="N", help="max commits to show (default: [git].log_limit)")
    diff_cmd = sub.add_parser("diff", help="show what a commit changed (diff vs parent)")
    diff_cmd.add_argument("commit", help="full or abbreviated commit sha")
    show_cmd = sub.add_parser("show", help="show one commit (message + patch)")
    show_cmd.add_argument("commit", help="full or abbreviated commit sha")
    revert_cmd = sub.add_parser("revert", help="roll back one session commit with a revert commit")
    revert_cmd.add_argument("commit", help="full or abbreviated commit sha")
    sub.add_parser("compact", help="prune superseded memory entries and create a compaction commit")
    sub.add_parser("gc", help="alias for compact")
    sub.add_parser("serve", help="run the MCP stdio server (JSON-RPC over stdin/stdout)")
    sub.add_parser("web", help="serve the memgit web UI (history, patches, one-click revert)")
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
        print(str(exc), file=sys.stderr)
    command = args.command
    if command == "doctor":
        return run_doctor(cfg)
    if cfg is None:
        return 2
    if command == "chat":
        return cmd_chat(cfg, args)
    if command == "recall":
        return cmd_recall(cfg, args)
    if command == "log":
        return cmd_log(cfg, args)
    if command == "diff":
        return cmd_diff(cfg, args)
    if command == "show":
        return cmd_show(cfg, args)
    if command == "revert":
        return cmd_revert(cfg, args)
    if command in ("compact", "gc"):
        return cmd_compact(cfg, args)
    if command == "serve":
        return cmd_serve(cfg, args)
    if command == "web":
        return cmd_web(cfg, args)
    print(f"unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
