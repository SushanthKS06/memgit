from __future__ import annotations

import json
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator


class ConfigError(ValueError):
    pass


def package_root() -> Path:
    return Path(__file__).resolve().parent


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str
    api_key_env: str
    models_endpoint: str
    chat_endpoint: str
    auth_header_name: str
    auth_header_template: str
    model: str
    judge_model: str
    model_preference_patterns: List[str]
    timeout_s: float
    max_retries: int
    backoff_s: float
    temperature: float
    judge_temperature: float
    max_calls: int
    json_retries: int

    @model_validator(mode="after")
    def _ranges(self) -> "LLMSettings":
        if self.max_retries < 0:
            raise ValueError("llm.max_retries must be >= 0")
        if self.backoff_s < 0:
            raise ValueError("llm.backoff_s must be >= 0")
        if self.timeout_s <= 0:
            raise ValueError("llm.timeout_s must be > 0")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("llm.temperature must be within [0, 2]")
        if not 0.0 <= self.judge_temperature <= 2.0:
            raise ValueError("llm.judge_temperature must be within [0, 2]")
        if self.max_calls <= 0:
            raise ValueError("llm.max_calls must be > 0")
        if self.json_retries < 0:
            raise ValueError("llm.json_retries must be >= 0")
        if not self.api_key_env.strip():
            raise ValueError("llm.api_key_env must be a non-empty env var NAME")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", self.api_key_env):
            raise ValueError("llm.api_key_env must be an env var NAME (UPPER_SNAKE), not a secret value")
        if "{key}" not in self.auth_header_template:
            raise ValueError("llm.auth_header_template must contain {key}")
        return self


class PromptsSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    consolidate: str
    few_shots: str
    fix_json: str
    recall_select: str
    recall_block: str
    answer: str
    commit_message: str

    @model_validator(mode="after")
    def _names(self) -> "PromptsSettings":
        for field_name, value in self.model_dump().items():
            if not value.strip():
                raise ValueError(f"prompts.{field_name} must be non-empty")
        return self


class MemorySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: str
    prompts_dir: str
    state_dir: str
    counter_file: str
    index_file: str
    header_prefix: str
    active_heading: str
    superseded_heading: str
    bullet: str
    id_open: str
    id_close: str
    id_prefix: str
    field_sep: str
    source_prefix: str
    turn_prefix: str
    quote_char: str
    superseded_by_prefix: str
    index_line_template: str
    max_ops_per_session: int
    allowed_extensions: List[str]
    max_path_depth: int
    slug_strip_pattern: str
    max_recall_files: int
    max_recall_tokens: int
    empty_memory_note: str
    token_count_method: str

    @field_validator("field_sep", "active_heading", "superseded_heading", "id_prefix", "quote_char")
    @classmethod
    def _sep(cls, v: str, info) -> str:
        if not v:
            raise ValueError(f"memory.{info.field_name} must be non-empty")
        return v

    @model_validator(mode="after")
    def _ranges(self) -> "MemorySettings":
        if self.max_ops_per_session <= 0:
            raise ValueError("memory.max_ops_per_session must be > 0")
        if self.max_path_depth <= 1:
            raise ValueError("memory.max_path_depth must be > 1")
        if not self.allowed_extensions:
            raise ValueError("memory.allowed_extensions must not be empty")
        if self.max_recall_files <= 0:
            raise ValueError("memory.max_recall_files must be > 0")
        if self.max_recall_tokens <= 0:
            raise ValueError("memory.max_recall_tokens must be > 0")
        if self.token_count_method not in ("words", "chars4"):
            raise ValueError("memory.token_count_method must be 'words' or 'chars4'")
        return self


class VerifySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_quote_chars: int
    dup_similarity: float
    normalise_rules: List[str]

    @field_validator("normalise_rules")
    @classmethod
    def _rules(cls, v: List[str]) -> List[str]:
        allowed = {"lowercase", "collapse_ws", "strip_edge_punct"}
        unknown = [r for r in v if r not in allowed]
        if unknown:
            raise ValueError(f"verify.normalise_rules contains unsupported rules: {unknown}")
        return v

    @model_validator(mode="after")
    def _ranges(self) -> "VerifySettings":
        if self.min_quote_chars < 1:
            raise ValueError("verify.min_quote_chars must be >= 1")
        if not 0.0 < self.dup_similarity <= 1.0:
            raise ValueError("verify.dup_similarity must be in (0, 1]")
        return self


class RecallSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fallback_grep: bool
    grep_min_token: int

    @field_validator("grep_min_token")
    @classmethod
    def _grep_min(cls, v: int) -> int:
        if v < 1:
            raise ValueError("recall.grep_min_token must be >= 1")
        return v


class GitSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    author_name: str
    author_email: str
    timeout_s: float
    core_autocrlf: bool
    log_limit: int

    @field_validator("timeout_s")
    @classmethod
    def _timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("git.timeout_s must be > 0")
        return v

    @field_validator("log_limit")
    @classmethod
    def _log_limit(cls, v: int) -> int:
        if v < 1:
            raise ValueError("git.log_limit must be >= 1")
        return v

    @model_validator(mode="after")
    def _names(self) -> "GitSettings":
        if not self.author_name.strip() or not self.author_email.strip():
            raise ValueError("git.author_name and git.author_email must be non-empty")
        return self


class WebSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str
    port: int
    page_size: int
    rejections_rows: int

    @field_validator("port")
    @classmethod
    def _port(cls, v: int) -> int:
        if not 0 <= v <= 65535:
            raise ValueError("web.port must be within [0, 65535] (0 = bind a free ephemeral port)")
        return v

    @field_validator("page_size", "rejections_rows")
    @classmethod
    def _positive(cls, v: int, info) -> int:
        if v < 1:
            raise ValueError(f"web.{info.field_name} must be >= 1")
        return v


class McpSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server_name: str


class LoggingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    usage_log_path: str
    rejections_log_path: str
    env_file: str

    @model_validator(mode="after")
    def _paths(self) -> "LoggingSettings":
        for field_name, value in (("usage_log_path", self.usage_log_path), ("rejections_log_path", self.rejections_log_path), ("env_file", self.env_file)):
            if not value.strip():
                raise ValueError(f"logging.{field_name} must be non-empty")
        return self


class BenchSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenarios_dir: str
    transcripts_dir: str
    runs_per_scenario: int
    output_dir: str
    abstain_phrases_file: str
    transcript_seed: int
    viability_memory_dir: str
    viability_transcripts: int
    viability_min_json_valid: float
    viability_min_verified: float

    @model_validator(mode="after")
    def _ranges(self) -> "BenchSettings":
        if self.runs_per_scenario <= 0:
            raise ValueError("bench.runs_per_scenario must be > 0")
        if self.viability_transcripts <= 0:
            raise ValueError("bench.viability_transcripts must be > 0")
        if not 0.0 < self.viability_min_json_valid <= 1.0:
            raise ValueError("bench.viability_min_json_valid must be in (0, 1]")
        if not 0.0 < self.viability_min_verified <= 1.0:
            raise ValueError("bench.viability_min_verified must be in (0, 1]")
        return self


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm: LLMSettings
    prompts: PromptsSettings
    memory: MemorySettings
    verify: VerifySettings
    recall: RecallSettings
    git: GitSettings
    web: WebSettings
    mcp: McpSettings
    logging: LoggingSettings
    bench: BenchSettings


@dataclass(frozen=True)
class LoadedConfig:
    settings: Settings
    layers: List[str]


def _coerce(value: str) -> Any:
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _deep_assign(root: Dict[str, Any], path: List[str], value: Any) -> None:
    node = root
    for part in path[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[path[-1]] = value


def _read_toml(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}")
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}")
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}")


def _env_layer(prefix: str = "MEMGIT_") -> Dict[str, Any]:
    layer: Dict[str, Any] = {}
    for name, raw in os.environ.items():
        if not name.startswith(prefix):
            continue
        if "__" not in name:
            # flat vars (e.g. the key-holding MEMGIT_API_KEY or MEMGIT_CONFIG_FILE) are not config
            continue
        path = name[len(prefix):].lower().split("__")
        if any(not part for part in path):
            continue
        _deep_assign(layer, path, _coerce(raw))
    return layer


def parse_overrides(pairs: List[str] | None) -> Dict[str, Any]:
    layer: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ConfigError(f"--set expects key=value, got: {pair!r}")
        key, raw = pair.split("=", 1)
        raw = raw.strip()
        if not raw:
            raise ConfigError(f"--set value missing, got: {pair!r}")
        path = key.strip().lower().split(".")
        if any(not part for part in path):
            raise ConfigError(f"--set key invalid: {pair!r}")
        _deep_assign(layer, path, _coerce(raw))
    return layer


def load_env_file(path: str | Path) -> List[str]:
    """Load a dotenv-style file into os.environ (setdefault; never overrides shell env).

    Returns a list of warning strings for malformed lines. File content is never logged.
    """
    warnings: List[str] = []
    env_path = Path(path)
    if not env_path.is_file():
        return warnings
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot read env file {env_path}: {exc}") from exc
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            warnings.append(f"{env_path.name}:{lineno}: missing '=', line skipped")
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if not key:
            warnings.append(f"{env_path.name}:{lineno}: empty key, line skipped")
            continue
        os.environ.setdefault(key, value)
    return warnings


def _find_file_layer(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        return Path(explicit)
    env_path = os.environ.get("MEMGIT_CONFIG_FILE", "")
    if env_path:
        return Path(env_path)
    candidate = Path.cwd() / "memgit.toml"
    if candidate.is_file():
        return candidate
    return None


def load_settings(
    config_file: Optional[str] = None,
    cli_overrides: Optional[List[str]] = None,
) -> LoadedConfig:
    merged = _read_toml(package_root() / "defaults.toml")
    layers = ["package:defaults.toml"]
    file_layer = _find_file_layer(config_file)
    if file_layer is not None:
        merged = _deep_update(merged, _read_toml(file_layer))
        layers.append(f"file:{file_layer}")
    env_layer = _env_layer()
    if env_layer:
        merged = _deep_update(merged, env_layer)
        layers.append("env:MEMGIT_*")
    cli_layer = parse_overrides(cli_overrides)
    if cli_layer:
        merged = _deep_update(merged, cli_layer)
        layers.append("cli:--set")
    try:
        settings = Settings.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(f"configuration invalid:\n{exc}") from exc
    return LoadedConfig(settings=settings, layers=layers)


def memory_dir(settings: Settings) -> Path:
    return Path(settings.memory.dir)


def prompts_dir(settings: Settings) -> Path:
    candidate = settings.memory.prompts_dir
    if candidate:
        return Path(candidate)
    return package_root() / "prompts"
