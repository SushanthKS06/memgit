"""Recovery benchmark: run scenarios x reps through the full session pipeline.

For every scenario file in [bench].scenarios_dir this harness:
  1. replays each session's transcript through MemorySession (consolidate ->
     verify -> apply -> git commit) on a wiped scratch store;
  2. for each probe, recalls memory, generates an answer, and classifies it;
  3. scores two recovery measures:
       reverified  - the expected fact is physically present in committed memory
       verbalized  - a judge model rates the answer as conveying the fact
plus abstain/wrong rates. Results go to bench/reports/benchmark.csv + .md;
no charts (per project rules). All thresholds live for viability only.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict
from typing import Literal

from bench.gen_data import generate_scenarios
from memgit.cli import bootstrap_settings
from memgit.config import ConfigError, LoadedConfig, Settings
from memgit.core.answer import Answerer
from memgit.core.consolidate import Turn
from memgit.core.llm import BudgetCounter, LLMClient, LLMError, LoadPrompt
from memgit.core.retrieve import Recaller
from memgit.core.session import MemorySession
from memgit.core.store import MemoryStore


class BenchError(RuntimeError):
    pass


class Probe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    expect: str


class SessionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str = "add"
    transcript: List[Turn]


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    seed: int
    sessions: List[SessionSpec]
    probes: List[Probe]


class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["correct", "wrong", "abstain"]
    reason: str


def load_scenarios(scenarios_dir: Path) -> List[Scenario]:
    if not scenarios_dir.is_dir():
        raise BenchError(
            f"scenarios dir {scenarios_dir} does not exist; "
            f"run 'python -m bench.gen_data --mode scenarios' to generate files"
        )
    paths = sorted(scenarios_dir.glob("scenario-*.json"))
    if not paths:
        raise BenchError(f"no scenario-*.json files in {scenarios_dir}; run bench/gen_data --mode scenarios")
    scenarios: List[Scenario] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            scenarios.append(Scenario.model_validate(payload))
        except (OSError, ValueError) as exc:
            raise BenchError(f"scenario {path.name} unreadable: {exc}") from exc
    return scenarios


def load_abstain_phrases(path: Path) -> List[str]:
    if not path.is_file():
        raise BenchError(f"abstain phrases file {path} missing; run bench/gen_data.py to generate it")
    payload = json.loads(path.read_text(encoding="utf-8"))
    phrases = payload.get("phrases")
    if not isinstance(phrases, list) or not phrases:
        raise BenchError(f"abstain phrases file {path} has no phrases")
    return [str(phrase).lower() for phrase in phrases]


def normalise(text: str) -> str:
    return " ".join(text.lower().split())


def _wipe_scratch(root: Path) -> None:
    """Windows-safe wipe: git marks .git/objects read-only; chmod + retry."""
    if not root.exists():
        return

    def on_readonly(func, path, exc) -> None:
        os.chmod(path, stat.S_IWRITE)
        func(path)

    shutil.rmtree(root, onexc=on_readonly)


def run_benchmark(
    cfg: LoadedConfig,
    scenarios: List[Scenario],
    abstain_phrases: List[str],
    llm_factory: Optional[Callable[[Settings], LLMClient]] = None,
) -> Dict:
    bench = cfg.settings.bench
    factory = llm_factory or (lambda settings: LLMClient(settings, budget=BudgetCounter()))
    client = factory(cfg.settings)
    recaller = Recaller(cfg.settings, client)
    answerer = Answerer(cfg.settings, client)
    judge_model = client.resolve_model("judge")

    rows: List[Dict] = []
    failed: List[str] = []
    for scenario in scenarios:
        for rep in range(1, bench.runs_per_scenario + 1):
            # wipe scratch memory per rep; counters restart, ids restart per run
            _wipe_scratch(Path(bench.benchmark_memory_dir))
            Path(bench.benchmark_memory_dir).mkdir(parents=True, exist_ok=True)
            rep_store = MemoryStore(cfg.settings, path=Path(bench.benchmark_memory_dir))
            rep_runner = MemorySession(cfg.settings, rep_store, client)
            for session in scenario.sessions:
                label = f"{scenario.id} rep{rep} sessions[{session.kind}]"
                try:
                    outcome = rep_runner.run(session.transcript)
                except LLMError as exc:
                    failed.append(f"{label}: {type(exc).__name__}: {exc}")
                    rep_store.finish_session()
                    continue
                if not outcome.ok:
                    failed.append(f"{label}: {outcome.error}")
                    continue
            active = [
                (rel, line)
                for rel, line in rep_store.active_entries()
            ]
            for probe_index, probe in enumerate(scenario.probes):
                row = {
                    "scenario_id": scenario.id,
                    "rep": rep,
                    "probe_index": probe_index,
                    "query": probe.query,
                    "expect": probe.expect,
                    "reverified": any(
                        normalise(probe.expect) in normalise(f"{line.fact} {line.quote}")
                        for _, line in active
                    ),
                    "answer": "",
                    "abstained": False,
                    "verdict": "",
                    "verdict_reason": "",
                    "wrong": False,
                }
                try:
                    block = recaller.recall(probe.query).block
                    answer = answerer.answer([Turn(turn=1, role="user", text=probe.query)], block)
                except LLMError as exc:
                    failed.append(f"{scenario.id} rep{rep} probe{probe_index}: {type(exc).__name__}: {exc}")
                    rep_store.finish_session()
                    break
                row["answer"] = answer
                answer_flat = normalise(answer)
                abstained = any(phrase in answer_flat for phrase in abstain_phrases)
                row["abstained"] = abstained
                if not abstained:
                    prompt = LoadPrompt(cfg.settings, cfg.settings.prompts.judge_recall).render(
                        expect=probe.expect,
                        query=probe.query,
                        answer=answer,
                    )
                    verdict, _ = client.chat_json(
                        [{"role": "user", "content": prompt}],
                        schema=JudgeVerdict,
                        purpose="judge_recall",
                        model=judge_model,
                        temperature=cfg.settings.llm.judge_temperature,
                    )
                    row["verdict"] = verdict.verdict
                    row["verdict_reason"] = verdict.reason
                    row["wrong"] = verdict.verdict == "wrong"
                rows.append(row)
    rates = _rates(rows)
    totals = client.totals()
    return {
        "scenarios_dir": str(bench.scenarios_dir),
        "benchmark_memory_dir": bench.benchmark_memory_dir,
        "runs_per_scenario": bench.runs_per_scenario,
        "scenario_count": len(scenarios),
        "abstain_phrases_file": str(bench.abstain_phrases_file),
        "rows": rows,
        "failed": failed,
        "results": {
            "probes_scored": len(rows),
            **rates,
            "llm_calls": totals["calls"],
            "llm_failures": totals["failures"],
            "prompt_tokens": totals["prompt_tokens"],
            "completion_tokens": totals["completion_tokens"],
        },
    }


def _rates(rows: List[Dict]) -> Dict[str, float]:
    count = len(rows)
    if count == 0:
        return {
            "reverified_rate": 0.0,
            "verbalized_rate": 0.0,
            "abstain_rate": 0.0,
            "wrong_rate": 0.0,
        }
    return {
        "reverified_rate": round(sum(1 for r in rows if r["reverified"]) / count, 4),
        "verbalized_rate": round(sum(1 for r in rows if r["verdict"] == "correct") / count, 4),
        "abstain_rate": round(sum(1 for r in rows if r["abstained"]) / count, 4),
        "wrong_rate": round(sum(1 for r in rows if r["wrong"]) / count, 4),
    }


def render_report(report: Dict) -> str:
    results = report["results"]
    lines = [
        "Memgit recovery benchmark",
        "=" * 40,
        f"scenarios:         {report['scenario_count']} x {report['runs_per_scenario']} runs "
        f"(probes scored: {results['probes_scored']})",
        f"scenarios dir:     {report['scenarios_dir']}",
        f"memory dir:        {report['benchmark_memory_dir']}",
        f"re-verified rate:  {results['reverified_rate']}",
        f"verbalized rate:   {results['verbalized_rate']}",
        f"abstain rate:      {results['abstain_rate']}",
        f"wrong rate:        {results['wrong_rate']}",
        f"LLM usage:         calls={results['llm_calls']} failures={results['llm_failures']} "
        f"prompt_tokens={results['prompt_tokens']} completion_tokens={results['completion_tokens']}",
    ]
    if report["failed"]:
        lines.append("failures:")
        lines.extend(f"  - {item}" for item in report["failed"])
    return "\n".join(lines)


def write_report(cfg: LoadedConfig, report: Dict) -> List[Path]:
    out_dir = Path(cfg.settings.bench.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "benchmark.csv"
    md_path = out_dir / "benchmark.md"
    keys = [
        "scenario_id",
        "rep",
        "probe_index",
        "query",
        "expect",
        "reverified",
        "abstained",
        "verdict",
        "verdict_reason",
        "wrong",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow(row)
    md_path.write_text(render_report(report) + "\n", encoding="utf-8")
    return [csv_path, md_path]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="memgit-bench", description="Recovery benchmark over scenario files")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=None)
    args = parser.parse_args(argv)
    try:
        cfg = bootstrap_settings(args.config_file, args.overrides)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    bench = cfg.settings.bench
    scenarios_path = Path(bench.scenarios_dir)
    if not scenarios_path.is_dir() or not list(scenarios_path.glob("scenario-*.json")):
        generated = generate_scenarios(args.config_file, args.overrides, None)
        print(f"generated scenarios: {generated}")
    try:
        scenarios = load_scenarios(scenarios_path)
        phrases = load_abstain_phrases(Path(bench.abstain_phrases_file))
        report = run_benchmark(cfg, scenarios, phrases)
    except (BenchError, LLMError) as exc:
        print(f"benchmark blocked by infrastructure error:\n{exc}", file=sys.stderr)
        print("hint: run 'memgit doctor' to diagnose config/api issues", file=sys.stderr)
        return 2
    print(render_report(report))
    for path in write_report(cfg, report):
        print(f"report written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
