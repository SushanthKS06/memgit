"""Generates synthetic transcript data files for bench/viability.py.

Content lives in the generated JSON data file, not in the runner logic.
The generator is intentionally dumb: seeded pools + templates.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memgit.cli import bootstrap_settings


FAMILY_EDITORS = ["VS Code", "Zed", "Neovim", "WebStorm", "Sublime Text", "Kate", "Helix", "RustRover"]
FAMILY_LANGS = ["Python", "Rust", "Go", "TypeScript", "Haskell", "Kotlin", "C#", "OCaml"]
FAMILY_CITIES = ["Lisbon", "Tallinn", "Osaka", "Montevideo", "Krakow", "Wellington", "Cork", "Tbilisi"]
FAMILY_ALLERGENS = ["hazelnuts", "celery", "sesame", "shellfish", "lupin", "mustard", "sulfites"]
FAMILY_PETS = [("Asphalt", "beagle"), ("Cobol", "corgi"), ("Kernel", "shepherd"), ("Pixel", "shiba inu"),
               ("Mango", "whippet"), ("Bazel", "samoyed"), ("Turbo", "dachshund"), ("Nacho", "corgi")]
FAMILY_PROJECTS = ["payments", "search", "onboarding", "billing", "ingest", "auth", "exporter", "cronos"]
FAMILY_STACKS = ["Postgres + Redis", "SQLite + NAS", "DuckDB", "MongoDB", "CockroachDB", "MariaDB + Memcached", " Cassandra", "EventStore"]

NOISE_USER = ["hey what's up", "lol ok", "hmm interesting", "what time is it", "anyway", "thanks, that helps", "ok cool"]
ACKS = ["Noted.", "Got it, recorded.", "Understood, I'll keep that in mind.", "Sounds good, logged.", "Okay, I'll remember that."]


def fact_sentence(kind: str, value: str) -> str:
    if kind == "editor":
        return f"My main editor is {value}; treat that as my default for all code work."
    if kind == "language":
        return f"For serious backend work my language of choice is {value}, no exceptions."
    if kind == "city":
        return f"I currently live in {value}, so schedule anything to that timezone."
    if kind == "allergy":
        return f"Important: I am allergic to {value}; never suggest recipes containing it."
    if kind == "project_stack":
        project, stack = value
        return f"We decided the {project} rewrite will use the {stack} stack."
    if kind == "pet":
        name, breed = value
        return f"My dog is called {name} and he is a {breed}."
    raise ValueError(f"unknown fact kind: {kind}")


def contradict_sentence(kind: str, old: str, new: str) -> str:
    if kind == "editor":
        return f"Update from today: I've stopped using {old} as my main editor; {new} is my main editor now."
    if kind == "language":
        return f"Change of plans: my backend language of choice is no longer {old}; it is {new} from now on."
    if kind == "city":
        return f"FYI I no longer live in {old}; I moved and now live in {new}."
    raise ValueError(f"kind {kind} cannot be contradicted")


def build_sessions(rng: random.Random, total: int) -> List[Dict[str, Any]]:
    sessions: List[Dict[str, Any]] = []
    stock: List[Tuple[str, Any]] = []
    used: Dict[str, set] = {}
    counters = {"editor": _cycler(rng, FAMILY_EDITORS), "language": _cycler(rng, FAMILY_LANGS),
                "city": _cycler(rng, FAMILY_CITIES), "allergy": _cycler(rng, FAMILY_ALLERGENS),
                "project_stack": None, "pet": _cycler(rng, FAMILY_PETS)}
    stack_pairs: List[Tuple[str, str]] = [(p, s.strip()) for p in FAMILY_PROJECTS for s in FAMILY_STACKS]
    rng.shuffle(stack_pairs)
    stack_iter = iter(stack_pairs)
    turn_no = 0

    def turns_of(user_texts: List[str]) -> List[Dict[str, Any]]:
        nonlocal turn_no
        rendered: List[Dict[str, Any]] = []
        for text in user_texts:
            if text.startswith("<ack>"):
                turn_no += 1
                rendered.append({"turn": turn_no, "role": "assistant", "text": text[5:]})
            else:
                turn_no += 1
                rendered.append({"turn": turn_no, "role": "user", "text": text})
        return rendered

    emit = 0
    while emit < total:
        emit += 1
        turn_no = 0
        if emit % 5 == 0 and any(kind in ("editor", "language", "city") for kind, _ in stock):
            cand_kind, cand_old = rng.choice([(k, v) for k, v in stock if k in ("editor", "language", "city")])
            new_value = next(c for c in counters[cand_kind] if c != cand_old)
            sessions.append({
                "index": emit,
                "kind": "contradiction",
                "transcript": turns_of([
                    contradict_sentence(cand_kind, cand_old, new_value),
                    f"<ack>{rng.choice(ACKS)}",
                ]),
            })
            stock.remove((cand_kind, cand_old))
            stock.append((cand_kind, new_value))
            continue
        if emit % 7 == 0:
            sessions.append({
                "index": emit,
                "kind": "noise",
                "transcript": turns_of([
                    rng.choice(NOISE_USER), f"<ack>{rng.choice(ACKS)}",
                    rng.choice(NOISE_USER), f"<ack>{rng.choice(ACKS)}",
                ]),
            })
            continue
        picks = rng.sample(["editor", "language", "city", "allergy", "project_stack", "pet"], rng.randint(1, 3))
        lines: List[str] = [rng.choice(NOISE_USER), f"<ack>{rng.choice(ACKS)}"]
        for kind in picks:
            if kind == "project_stack":
                pair = next(stack_iter, None)
                if pair is None or pair in used.get("project_stack", set()):
                    continue
                used.setdefault("project_stack", set()).add(pair)
                lines.append(fact_sentence(kind, pair))
                stock.append(("project_stack", pair))
            else:
                value = next(c for c in counters[kind] if (kind, c) not in used.get(kind, set()))
                used.setdefault(kind, set()).add((kind, value))
                lines.append(fact_sentence(kind, value))
                stock.append((kind, value))
            lines.append(f"<ack>{rng.choice(ACKS)}")
        sessions.append({"index": emit, "kind": "add", "transcript": turns_of(lines)})
    return sessions


def _cycler(rng: random.Random, values: List[str]):
    order = list(values)
    rng.shuffle(order)
    while True:
        for value in order:
            yield value
        rng.shuffle(order)


def write_abstain_phrases(path: Path) -> None:
    phrases = [
        "i don't have that information",
        "i don't have that on record",
        "i don't know based on your memory",
        "nothing was stored about that",
        "i have no memory of that",
        "based on your saved memory, i cannot answer",
        "there is no note about that in memory",
        "not in your saved notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump({"phrases": phrases}, fh, indent=2)
        fh.write("\n")


PROBE_SENTENCES = {
    "editor": "quick check — what's my main editor again?",
    "language": "what language do I use for serious backend work?",
    "city": "where do I actually live right now?",
    "allergy": "remind me what I'm allergic to, I keep forgetting",
    "project_stack": "which stack did we pick for the {project} rewrite?",
    "pet": "what breed is {name} again?",
}

FAMILY_PROJECT_STACKS = ["Postgres + Redis", "SQLite + NAS", "DuckDB", "MongoDB", "CockroachDB", "MariaDB + Memcached"]


def build_scenarios(rng: random.Random, count: int) -> List[Dict[str, Any]]:
    """One scenario = add-facts session(s) + a probe question per emitted fact."""
    scenarios: List[Dict[str, Any]] = []
    for emit in range(1, count + 1):
        kinds = rng.sample(["editor", "language", "city", "allergy", "project_stack", "pet"], rng.randint(2, 3))
        values: Dict[str, Any] = {}
        editor = rng.choice(FAMILY_EDITORS)
        language = rng.choice(FAMILY_LANGS)
        city = rng.choice(FAMILY_CITIES)
        allergen = rng.choice(FAMILY_ALLERGENS)
        pet = rng.choice(FAMILY_PETS)
        project = rng.choice(FAMILY_PROJECTS)
        stack = rng.choice(FAMILY_STACKS)
        probe_texts: List[Dict[str, str]] = []
        lines: List[str] = [rng.choice(NOISE_USER)]
        for kind in kinds:
            if kind == "editor":
                values["editor"] = editor
                lines.append(fact_sentence("editor", editor))
                probe_texts.append({"fingerprint": PROBE_SENTENCES["editor"], "expect": editor})
            elif kind == "language":
                values["language"] = language
                lines.append(fact_sentence("language", language))
                probe_texts.append({"fingerprint": PROBE_SENTENCES["language"], "expect": language})
            elif kind == "city":
                values["city"] = city
                lines.append(fact_sentence("city", city))
                probe_texts.append({"fingerprint": PROBE_SENTENCES["city"], "expect": city})
            elif kind == "allergy":
                values["allergy"] = allergen
                lines.append(fact_sentence("allergy", allergen))
                probe_texts.append({"fingerprint": PROBE_SENTENCES["allergy"], "expect": allergen})
            elif kind == "project_stack":
                values[f"project:{project}"] = stack
                lines.append(fact_sentence("project_stack", (project, stack)))
                probe_texts.append({"fingerprint": PROBE_SENTENCES["project_stack"].format(project=project), "expect": stack})
            elif kind == "pet":
                name, breed = pet
                values[f"pet:{name}"] = breed
                lines.append(fact_sentence("pet", pet))
                probe_texts.append({"fingerprint": PROBE_SENTENCES["pet"].format(name=name), "expect": breed})
            lines.append(f"<ack>{rng.choice(ACKS)}")
        turn_no = 0
        transcript: List[Dict[str, Any]] = []
        for text in lines:
            if text.startswith("<ack>"):
                turn_no += 1
                transcript.append({"turn": turn_no, "role": "assistant", "text": text[5:]})
            else:
                turn_no += 1
                transcript.append({"turn": turn_no, "role": "user", "text": text})
        scenarios.append(
            {
                "id": f"scenario-{emit:02d}",
                "seed": rng.randint(0, 2**31 - 1),
                "sessions": [
                    {"kind": "add", "transcript": transcript},
                    {
                        "kind": "noise",
                        "transcript": [
                            {"turn": 1, "role": "user", "text": rng.choice(NOISE_USER)},
                            {"turn": 2, "role": "assistant", "text": rng.choice(ACKS)},
                        ],
                    },
                ],
                "probes": [
                    {
                        "query": probe["fingerprint"],
                        "expect": probe["expect"],
                    }
                    for probe in probe_texts
                ],
            }
        )
    return scenarios


def generate_scenarios(config_file: Optional[str], overrides: List[str], seed: Optional[int]) -> Path:
    cfg = bootstrap_settings(config_file, overrides)
    bench = cfg.settings.bench
    out_dir = Path(bench.scenarios_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    used_seed = seed if seed is not None else bench.transcript_seed
    rng = random.Random(used_seed)
    for old in sorted(out_dir.glob("scenario-*.json")):
        old.unlink()
    count = max(1, bench.viability_transcripts // 2)
    for scenario in build_scenarios(rng, count):
        path = out_dir / f"{scenario['id']}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(scenario, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    return out_dir


def generate(config_file: Optional[str], overrides: List[str], data_file: Optional[str], seed: Optional[int]) -> Path:
    cfg = bootstrap_settings(config_file, overrides)
    bench = cfg.settings.bench
    transcripts_dir = Path(bench.transcripts_dir)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    target = Path(data_file) if data_file else transcripts_dir / "viability_transcripts.json"
    used_seed = seed if seed is not None else bench.transcript_seed
    rng = random.Random(used_seed)
    payload = {
        "seed": used_seed,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sessions": build_sessions(rng, bench.viability_transcripts),
    }
    with target.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    write_abstain_phrases(Path(bench.abstain_phrases_file))
    return target


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="memgit-gen-data", description="Generate bench data files and scenarios")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", choices=["viability", "scenarios"], default="viability")
    args = parser.parse_args(argv)
    try:
        if args.mode == "scenarios":
            target = generate_scenarios(args.config_file, args.overrides, args.seed)
        else:
            target = generate(args.config_file, args.overrides, args.data_file, args.seed)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
