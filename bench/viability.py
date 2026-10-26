"""Phase 0 viability gate: run consolidation on the synthetic transcript data file.

Usage: memgit-bench-viability [--regen]
Pass thresholds: bench.viability_min_json_valid, bench.viability_min_verified.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from memgit.cli import bootstrap_settings
from memgit.config import ConfigError, LoadedConfig
from memgit.core.consolidate import Consolidator, Turn
from memgit.core.llm import BudgetCounter, LLMClient, LLMError
from memgit.core.store import MemoryStore
from memgit.core.verify import Verifier


def load_transcripts(path: Path) -> List[Dict]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except OSError as exc:
        raise LLMError(f"cannot read transcript data file {path}: {exc}") from exc
    except ValueError as exc:
        raise LLMError(f"transcript data file {path} is not valid JSON: {exc}") from exc
    sessions = payload.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise LLMError(
            f"transcript data file {path} has no sessions; run bench/gen_data.py to generate it"
        )
    return sessions


def to_turns(raw_transcript: List[Dict]) -> List[Turn]:
    return [
        Turn(turn=item["turn"], role=item["role"], text=item["text"])
        for item in raw_transcript
    ]


def run_gate(
    cfg: LoadedConfig,
    transcripts_path: Path,
) -> Dict:
    bench = cfg.settings.bench
    memory_root = Path(bench.viability_memory_dir)
    if memory_root.exists():
        shutil.rmtree(memory_root)
    memory_root.mkdir(parents=True, exist_ok=True)
    sessions = load_transcripts(transcripts_path)
    client = LLMClient(cfg.settings, budget=BudgetCounter())
    store = MemoryStore(cfg.settings, path=memory_root)
    consolidator = Consolidator(cfg.settings, client)
    rejected_by_reason: Counter = Counter()
    total_sessions = len(sessions)
    consolidated = 0
    failed_sessions: List[str] = []
    retry_sessions = 0
    accepted_adds = 0
    accepted_supersedes = 0
    noop_verdicts = 0
    non_consolidated = 0

    for session in sessions:
        session_label = f"session {session.get('index', len(failed_sessions) + 1)}"
        try:
            store.begin_session()
            consolidator.model
            result = consolidator.propose(
                transcript=to_turns(session["transcript"]),
                active_entries=store.active_entries(),
                next_id_example=store.next_id_example(),
            )
        except LLMError as exc:
            non_consolidated += 1
            failed_sessions.append(f"{session_label}: {exc}")
            store.session_id = None
            continue
        consolidated += 1
        if result.used_retry:
            retry_sessions += 1
        verifier = Verifier(cfg.settings, turns=to_turns(session["transcript"]), active_entries=store.active_entries())
        summary = verifier.verify_all(result.ops)
        accepted, rejected, noops = summary.counts()
        accepted_adds += sum(1 for v in summary.verdicts if v.status == "accepted_add")
        accepted_supersedes += sum(1 for v in summary.verdicts if v.status == "accepted_supersede")
        noop_verdicts += noops
        for verdict in summary.verdicts:
            if verdict.status == "rejected":
                reason = verdict.reason.split(":")[0] if ":" in verdict.reason else verdict.reason
                rejected_by_reason[reason] += 1
        store.apply(summary.verdicts)

    rejected_total = sum(rejected_by_reason.values())
    candidates = accepted_adds + accepted_supersedes + rejected_total
    verified_rate = 1.0 if candidates == 0 else (accepted_adds + accepted_supersedes) / candidates
    json_valid_rate = consolidated / total_sessions if total_sessions else 0.0
    totals = client.totals()
    report = {
        "transcripts_file": str(transcripts_path),
        "memory_dir": str(memory_root),
        "sessions_total": total_sessions,
        "sessions_consolidated": consolidated,
        "sessions_not_consolidated": non_consolidated,
        "sessions_used_json_retry": retry_sessions,
        "json_valid_rate": round(json_valid_rate, 4),
        "verified_op_rate": round(verified_rate, 4),
        "ops_accepted_add": accepted_adds,
        "ops_accepted_supersede": accepted_supersedes,
        "ops_rejected_total": rejected_total,
        "rejected_by_reason": dict(rejected_by_reason),
        "oops_noops": noop_verdicts,
        "llm_calls": totals["calls"],
        "llm_failures": totals["failures"],
        "prompt_tokens": totals["prompt_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "usage_log": cfg.settings.logging.usage_log_path,
        "thresholds": {
            "min_json_valid": bench.viability_min_json_valid,
            "min_verified": bench.viability_min_verified,
        },
        "failed_sessions": failed_sessions,
        "passed": json_valid_rate >= bench.viability_min_json_valid
        and verified_rate >= bench.viability_min_verified,
    }
    return report


def render_report(report: Dict) -> str:
    lines = [
        "Memgit viability gate",
        "=" * 40,
        f"transcripts file:  {report['transcripts_file']}",
        f"memory dir:        {report['memory_dir']}",
        f"consolidated:      {report['sessions_consolidated']}/{report['sessions_total']}",
        f"  used json retry: {report['sessions_used_json_retry']}",
        f"  not consolidated:{report['sessions_not_consolidated']}",
        f"op counts:         add={report['ops_accepted_add']} supersede={report['ops_accepted_supersede']} "
        f"rejected={report['ops_rejected_total']} noop={report['oops_noops']}",
        f"rejected reasons:  {json.dumps(report['rejected_by_reason'])}",
        f"LLM usage:         calls={report['llm_calls']} failures={report['llm_failures']} "
        f"prompt_tokens={report['prompt_tokens']} completion_tokens={report['completion_tokens']}",
        f"usage log:         {report['usage_log']}",
        f"json-valid rate:   {report['json_valid_rate']} (min {report['thresholds']['min_json_valid']})",
        f"verified-op rate:  {report['verified_op_rate']} (min {report['thresholds']['min_verified']})",
        f"verdict:           {'PASS' if report['passed'] else 'FAIL'}",
    ]
    if report["failed_sessions"]:
        lines.append("failed sessions:")
        lines.extend(f"  - {item}" for item in report["failed_sessions"])
    return "\n".join(lines)


def write_report(cfg: LoadedConfig, report: Dict) -> Path:
    out_dir = Path(cfg.settings.bench.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "viability.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
        fh.write("\n")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="memgit-viability", description="Phase 0 viability gate")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--regen", action="store_true", help="regenerate the synthetic transcript file")
    args = parser.parse_args(argv)
    try:
        cfg = bootstrap_settings(args.config_file, args.overrides)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    bench = cfg.settings.bench
    transcripts_dir = Path(bench.transcripts_dir)
    transcripts_path = Path(args.data_file) if args.data_file else transcripts_dir / "viability_transcripts.json"
    if args.regen or not transcripts_path.is_file():
        from bench.gen_data import generate

        generated = generate(args.config_file, args.overrides, args.data_file, None)
        print(f"generated transcripts: {generated}")
    try:
        report = run_gate(cfg, transcripts_path)
    except LLMError as exc:
        print(f"viability gate blocked by LLM infrastructure error:\n{exc}", file=sys.stderr)
        print("hint: run 'memgit doctor' to diagnose config/api issues", file=sys.stderr)
        return 2
    print(render_report(report))
    path = write_report(cfg, report)
    print(f"report written: {path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
