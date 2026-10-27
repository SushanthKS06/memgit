import json
from pathlib import Path

import httpx
import pytest

from memgit.config import LoadedConfig
from memgit.core.llm import BudgetCounter, LLMClient
from bench.report import (
    BenchError,
    load_abstain_phrases,
    load_scenarios,
    run_benchmark,
    write_report,
)

ADD_CONTENT = json.dumps(
    {
        "ops": [
            {
                "op": "add",
                "file": "prefs.md",
                "text": "The user's main editor is Neovim",
                "quote": "My main editor is Neovim",
                "turn": 2,
                "supersedes": None,
                "confidence": 0.91,
            }
        ]
    }
)

SCEN1 = {
    "id": "scenario-01",
    "seed": 422,
    "sessions": [
        {
            "kind": "add",
            "transcript": [
                {"turn": 1, "role": "assistant", "text": "Noted up front."},
                {"turn": 2, "role": "user", "text": "FYI My main editor is Neovim as of this week, honestly."},
            ],
        },
    ],
    "probes": [{"query": "what is my main editor again?", "expect": "Neovim"}],
}


class ChatMock:
    """FIFO chat queue backed by /models discovery; contents prepared per call order."""

    def __init__(self, models, contents: list[str]) -> None:
        self.models = models
        self.queue = list(contents)

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": m} for m in self.models]})
        if path.endswith("/chat/completions"):
            payload = json.loads(request.content.decode("utf-8"))
            content = self.queue.pop(0)
            return httpx.Response(
                200,
                json={
                    "id": "mock",
                    "object": "chat.completion",
                    "model": payload["model"],
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 4},
                },
            )
        return httpx.Response(404, json={"error": path})


@pytest.fixture()
def bench_env(settings_factory, api_key_env, tmp_path: Path):
    def build(scenarios: list[dict], contents_pool: list[str], runs: int = 1):
        scenarios_dir = tmp_path / "scens"
        scenarios_dir.mkdir(parents=True)
        for scenario in scenarios:
            (scenarios_dir / f"{scenario['id']}.json").write_text(json.dumps(scenario), encoding="utf-8")
        phrases_path = tmp_path / "abstains.json"
        phrases_path.write_text(json.dumps({"phrases": ["i don't have that information", "nothing was stored about that"]}), encoding="utf-8")
        overrides = [
            f"bench.scenarios_dir={scenarios_dir}",
            f"bench.abstain_phrases_file={phrases_path}",
            f"bench.runs_per_scenario={runs}",
            f"bench.benchmark_memory_dir={tmp_path / 'benchmem'}",
            f"bench.output_dir={tmp_path / 'reports'}",
            f"logging.usage_log_path={tmp_path / 'usage.jsonl'}",
            f"logging.rejections_log_path={tmp_path / 'rejections.jsonl'}",
            "llm.base_url=http://testserver/v1",
            "llm.max_retries=0",
            "llm.backoff_s=0",
        ]
        s = settings_factory(*overrides)
        cfg = LoadedConfig(settings=s, layers=["defaults", "cli"])
        mock = ChatMock(["mini-west", "large-east"], contents_pool * runs)
        client = LLMClient(
            s,
            budget=BudgetCounter(),
            http_client=httpx.Client(transport=httpx.MockTransport(mock.handler), timeout=s.llm.timeout_s),
        )
        return cfg, mock, client, s

    return build


def test_load_and_missing_files(settings_factory, tmp_path: Path):
    missing = tmp_path / "nope"
    with pytest.raises(BenchError) as missing_error:
        load_scenarios(missing)
    assert "scenarios dir" in str(missing_error.value)
    with pytest.raises(BenchError) as phrase_error:
        load_abstain_phrases(tmp_path / "missing.json")
    assert "abstain phrases" in str(phrase_error.value)


def test_benchmark_roundtrip(bench_env):
    per_rep_pool = [
        ADD_CONTENT,                              # session consolidation
        json.dumps({"files": ["prefs.md"]}),      # probe select
        "you told me your main editor is Neovim.",  # answer
        json.dumps({"verdict": "correct", "reason": "Re-verified."}),
    ]
    cfg, mock, client, s = bench_env([SCEN1], per_rep_pool, runs=2)
    scenarios = load_scenarios(Path(s.bench.scenarios_dir))
    phrases = load_abstain_phrases(Path(s.bench.abstain_phrases_file))
    report = run_benchmark(cfg, scenarios, phrases, llm_factory=lambda _settings: client)
    assert report["results"]["probes_scored"] == 2  # 1 probe x 2 reps
    assert report["results"]["reverified_rate"] == 1.0
    assert report["results"]["verbalized_rate"] == 1.0
    assert report["results"]["abstain_rate"] == 0.0
    assert report["failed"] == []
    csv_path, md_path = write_report(cfg, report)
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "Neovim" in csv_text
    md_lines = md_path.read_text(encoding="utf-8")
    assert "re-verified rate:  1.0" in md_lines
    assert "verbalized rate:   1.0" in md_lines
    csv_text = (Path(s.bench.output_dir) / "benchmark.csv").read_text(encoding="utf-8")
    assert "scenario-01,1,0" in csv_text
    assert "True,False,correct" in csv_text


def test_benchmark_abstain_skips_judge(bench_env):
    # empty ops -> no memory files -> recall skips the LLM select entirely;
    # call order: consolidation, answer (judge skipped due to abstain phrase)
    per_rep_pool = [
        json.dumps({"ops": []}),
        "sorry, i don't have that information.",
    ]
    cfg, mock, client, s = bench_env([SCEN1], per_rep_pool, runs=1)
    scenarios = load_scenarios(Path(s.bench.scenarios_dir))
    phrases = load_abstain_phrases(Path(s.bench.abstain_phrases_file))
    report = run_benchmark(cfg, scenarios, phrases, llm_factory=lambda _settings: client)
    row = report["rows"][0]
    assert row["abstained"] is True
    assert row["verdict"] == ""  # judge skipped for abstained answers
    assert row["reverified"] is False
    assert report["results"]["abstain_rate"] == 1.0
    assert report["results"]["verbalized_rate"] == 0.0


def test_benchmark_wrong_verdict_counts(bench_env):
    per_rep_pool = [
        ADD_CONTENT,
        json.dumps({"files": ["prefs.md"]}),
        "I believe your main editor is VS Code.",
        json.dumps({"verdict": "wrong", "reason": "expects Neovim"}),
    ]
    cfg, mock, client, s = bench_env([SCEN1], per_rep_pool, runs=1)
    scenarios = load_scenarios(Path(s.bench.scenarios_dir))
    phrases = load_abstain_phrases(Path(s.bench.abstain_phrases_file))
    report = run_benchmark(cfg, scenarios, phrases, llm_factory=lambda _settings: client)
    assert report["results"]["wrong_rate"] == 1.0
    assert report["results"]["verbalized_rate"] == 0.0
    assert report["results"]["reverified_rate"] == 1.0


def test_benchmark_session_failure_recorded(bench_env):
    # consolidation fails after fix retry (2 invalid contents); probes then still
    # run on empty memory: no select (no files), answer abstains on empty block
    per_rep_pool = [
        "not json at all",
        "```still not json```",
        "sorry, i don't have that information.",
    ]
    cfg, mock, client, s = bench_env([SCEN1], per_rep_pool, runs=1)
    scenarios = load_scenarios(Path(s.bench.scenarios_dir))
    phrases = load_abstain_phrases(Path(s.bench.abstain_phrases_file))
    report = run_benchmark(cfg, scenarios, phrases, llm_factory=lambda _settings: client)
    assert report["failed"], "session failure must be reported"
    joined = "\n".join(report["failed"])
    assert "sessions[add]" in joined
    assert "failed JSON validation" in joined
    # probes still get scored on empty memory; abstain answer keeps judge out
    assert report["results"]["probes_scored"] == 1
    assert report["results"]["reverified_rate"] == 0.0


def test_viability_write_report_markdown_and_json(tmp_path, settings_factory):
    from bench.viability import write_report as viability_write_report

    s = settings_factory(f"bench.output_dir={tmp_path}")
    cfg = LoadedConfig(settings=s, layers=[])
    fake_report = {
        "transcripts_file": "test_transcripts.json",
        "memory_dir": "test_mem",
        "sessions_total": 5,
        "sessions_consolidated": 5,
        "sessions_not_consolidated": 0,
        "sessions_used_json_retry": 1,
        "json_valid_rate": 1.0,
        "verified_op_rate": 0.85,
        "ops_accepted_add": 3,
        "ops_accepted_supersede": 1,
        "ops_rejected_total": 1,
        "rejected_by_reason": {"quote_not_found": 1},
        "noop_verdicts": 0,
        "llm_calls": 6,
        "llm_failures": 0,
        "prompt_tokens": 1200,
        "completion_tokens": 400,
        "usage_log": "test_usage.jsonl",
        "thresholds": {"min_json_valid": 0.9, "min_verified": 0.8},
        "failed_sessions": [],
        "passed": True,
    }
    json_path, md_path = viability_write_report(cfg, fake_report)
    assert json_path.is_file()
    assert md_path.is_file()
    md_content = md_path.read_text(encoding="utf-8")
    assert "# Memgit Viability Report" in md_content
    assert "JSON-valid rate" in md_content
    assert "Verified-op rate" in md_content
    assert "quote_not_found" in md_content
    assert "PASS" in md_content

