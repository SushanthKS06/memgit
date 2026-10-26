import pytest

from memgit.cli import bootstrap_settings


@pytest.fixture()
def settings_factory(tmp_path, monkeypatch):
    """Builds a Settings with CLI-style overrides after chdir into an empty tmp cwd."""

    def factory(*overrides: str):
        monkeypatch.chdir(tmp_path)
        cfg = bootstrap_settings(None, list(overrides))
        return cfg.settings

    return factory


@pytest.fixture()
def api_key_env(monkeypatch):
    monkeypatch.setenv("MEMGIT_API_KEY", "test-key-not-secret")
