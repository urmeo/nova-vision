import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from novavision.revisions import resolve_model_revision

COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def hub(monkeypatch):
    client = SimpleNamespace(
        model_info=Mock(return_value=SimpleNamespace(sha=COMMIT)),
        try_to_load_from_cache=Mock(return_value=None),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", client)
    return client


@pytest.mark.parametrize("revision", [COMMIT, COMMIT.upper()])
def test_full_commit_does_not_require_hub(monkeypatch, revision):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    assert resolve_model_revision("owner/custom-model", revision) == COMMIT


@pytest.mark.parametrize("revision", [None, "main", "release", "v1.2", COMMIT[:7]])
def test_resolves_symbolic_revision_with_deadline(hub, revision):
    assert resolve_model_revision("owner/custom-model", revision) == COMMIT

    hub.model_info.assert_called_once_with(
        "owner/custom-model", revision=revision or "main", timeout=30
    )
    hub.try_to_load_from_cache.assert_not_called()


def test_normalizes_resolved_commit(hub):
    hub.model_info.return_value.sha = COMMIT.upper()

    assert resolve_model_revision("owner/custom-model", "release") == COMMIT


@pytest.mark.parametrize("commit", [None, "main", COMMIT[:7], "g" * 40, 123])
def test_rejects_invalid_hub_commit_without_using_cache(hub, commit):
    hub.model_info.return_value.sha = commit

    with pytest.raises(ValueError, match="invalid commit"):
        resolve_model_revision("owner/custom-model", "release")

    hub.try_to_load_from_cache.assert_not_called()


@pytest.mark.parametrize("filename", ["config.json", "model_index.json"])
def test_resolves_offline_reference_to_cached_commit(hub, tmp_path, filename):
    config = tmp_path / "models--owner--custom-model" / "snapshots" / COMMIT / filename
    config.parent.mkdir(parents=True)
    config.write_text("{}")
    hub.model_info.side_effect = TimeoutError("offline")
    hub.try_to_load_from_cache.side_effect = lambda model, name, **kwargs: (
        str(config) if name == filename else None
    )

    assert resolve_model_revision("owner/custom-model", "release") == COMMIT

    hub.try_to_load_from_cache.assert_any_call("owner/custom-model", filename, revision="release")


@pytest.mark.parametrize(
    "cached",
    [None, object(), "snapshots/main/config.json", f"local/{COMMIT}/config.json"],
)
def test_rejects_unverified_cache_entries(hub, tmp_path, cached):
    if isinstance(cached, str):
        config = tmp_path / cached
        config.parent.mkdir(parents=True)
        config.write_text("{}")
        cached = str(config)
    error = TimeoutError("offline")
    hub.model_info.side_effect = error
    hub.try_to_load_from_cache.return_value = cached

    with pytest.raises(RuntimeError, match="immutable commit") as failure:
        resolve_model_revision("owner/custom-model", "release")

    assert failure.value.__cause__ is error


def test_rejects_missing_cached_file(hub, tmp_path):
    hub.model_info.side_effect = TimeoutError("offline")
    hub.try_to_load_from_cache.return_value = str(tmp_path / "snapshots" / COMMIT / "config.json")

    with pytest.raises(RuntimeError, match="immutable commit"):
        resolve_model_revision("owner/custom-model", "release")


def test_resolves_real_hub_cache_reference_offline(monkeypatch, tmp_path):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from huggingface_hub import constants

    model_cache = tmp_path / "models--owner--custom-model"
    refs = model_cache / "refs"
    refs.mkdir(parents=True)
    (refs / "release").write_text(COMMIT)
    snapshot = model_cache / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path))
    monkeypatch.setattr(huggingface_hub, "model_info", Mock(side_effect=TimeoutError("offline")))

    assert resolve_model_revision("owner/custom-model", "release") == COMMIT


def test_missing_hub_dependency_has_install_instruction(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(RuntimeError, match=r"install .\[ml\]"):
        resolve_model_revision("owner/custom-model", None)


@pytest.mark.parametrize("revision", [None, COMMIT])
def test_rejects_local_model_directory_before_hub_lookup(hub, tmp_path, revision):
    with pytest.raises(ValueError, match="Local model paths"):
        resolve_model_revision(str(tmp_path), revision)

    hub.model_info.assert_not_called()


@pytest.mark.parametrize("model_id", ["./model", "../model", "~/model", "/missing-model"])
def test_rejects_explicit_local_paths(hub, model_id):
    with pytest.raises(ValueError, match="Local model paths"):
        resolve_model_revision(model_id, COMMIT)

    hub.model_info.assert_not_called()
