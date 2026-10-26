from pathlib import Path

import pytest

from memgit.config import ConfigError, load_settings


def test_defaults_load(settings_factory):
    s = settings_factory()
    assert isinstance(s.web.port, int)
    assert isinstance(s.llm.timeout_s, float)
    assert isinstance(s.verify.dup_similarity, float)
    assert isinstance(s.memory.allowed_extensions, list)
    assert s.mcp.server_name


def test_env_overrides_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "memgit.toml").write_text('[llm]\nmodel = "from-toml"\n', encoding="utf-8")
    monkeypatch.setenv("MEMGIT_LLM__MODEL", "from-env")
    cfg = load_settings(None, None)
    assert cfg.settings.llm.model == "from-env"
    assert [layer for layer in cfg.layers].count("env:MEMGIT_*") == 1
    monkeypatch.delenv("MEMGIT_LLM__MODEL")
    cfg = load_settings(None, None)
    assert cfg.settings.llm.model == "from-toml"


def test_cli_override_beats_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "memgit.toml").write_text('[llm]\nmodel = "from-toml"\n', encoding="utf-8")
    monkeypatch.setenv("MEMGIT_LLM__MODEL", "from-env")
    cfg = load_settings(None, ["llm.model=from-cli"])
    assert cfg.settings.llm.model == "from-cli"


def test_explicit_config_file(tmp_path):
    toml = tmp_path / "custom.toml"
    toml.write_text('[web]\nport = 9999\n', encoding="utf-8")
    cfg = load_settings(str(toml), None)
    assert cfg.settings.web.port == 9999


def test_override_coerces_numbers(settings_factory):
    s = settings_factory("web.port=6543", "llm.timeout_s=2.5")
    assert s.web.port == 6543
    assert s.llm.timeout_s == 2.5


def test_unknown_key_rejected(settings_factory):
    with pytest.raises(ConfigError):
        settings_factory("web.colour=red")


def test_missing_value_rejected(settings_factory):
    with pytest.raises(ConfigError):
        settings_factory("llm.model=")


def test_env_bad_json_stays_string(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMGIT_LLM__MODEL", "not json with spaces")
    cfg = load_settings(None, None)
    assert cfg.settings.llm.model == "not json with spaces"


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_settings(str(tmp_path / "nope.toml"), None)
