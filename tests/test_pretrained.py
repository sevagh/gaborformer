"""Downloader integrity, offline behavior, and CLI routing, without network or model inference."""
import json
from pathlib import Path
import subprocess
import sys

import huggingface_hub
import httpx
from huggingface_hub.errors import LocalEntryNotFoundError, RepositoryNotFoundError
import pytest

from gaborformer import pretrained
from scripts.prepare_huggingface import model_card


@pytest.fixture
def hub_files(tmp_path, monkeypatch):
    payload = b"small checkpoint fixture"
    path = tmp_path / "weights.safetensors"
    path.write_bytes(payload)
    artifact = {"filename": path.name, "sha256": pretrained.sha256(path)}
    monkeypatch.setattr(pretrained, "WEIGHTS", {"fp16": artifact})
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_type": "gaborformer", "weights": {"fp16": artifact}}))
    calls = []

    def download(*, filename, **kwargs):
        calls.append({"filename": filename, **kwargs})
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return path, config, calls


@pytest.mark.parametrize("offline", [False, True])
def test_download_checks_manifest_and_checkpoint(hub_files, offline):
    path, _, calls = hub_files
    revision = "a" * 40
    assert pretrained.download_weights(offline=offline, revision=revision) == path
    assert [c["filename"] for c in calls] == ["config.json", path.name]
    assert all(c["revision"] == revision and c["local_files_only"] is offline for c in calls)
    assert all(c["repo_id"] == "sevagh/gaborformer" for c in calls)


def test_corrupt_cached_checkpoint_is_rejected(hub_files):
    path, _, _ = hub_files
    path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        pretrained.download_weights()


@pytest.mark.parametrize("config", [[], {"model_type": "wrong"}, {"model_type": "gaborformer", "weights": None}])
def test_wrong_manifest_is_rejected_before_weights_download(hub_files, config):
    _, config_path, calls = hub_files
    config_path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="does not match"):
        pretrained.download_weights()
    assert len(calls) == 1


def test_offline_cache_miss_explains_local_alternative(monkeypatch):
    def missing(**kwargs):
        assert kwargs["local_files_only"] is True
        raise LocalEntryNotFoundError("cache miss")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    with pytest.raises(RuntimeError, match="--weights"):
        pretrained.download_weights(offline=True)


def test_private_or_unpublished_repo_has_actionable_error(monkeypatch):
    def missing(**kwargs):
        response = httpx.Response(404, request=httpx.Request("GET", "https://huggingface.co/sevagh/gaborformer"))
        raise RepositoryNotFoundError("not public", response=response)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    with pytest.raises(RuntimeError, match="hf auth login"):
        pretrained.download_weights()


def test_builtin_hashes_match_published_checksums():
    root = Path(__file__).resolve().parents[1]
    listed = dict(line.split()[::-1] for line in (root / "weights/SHA256SUMS").read_text().splitlines())
    assert listed == {a["filename"]: a["sha256"] for a in pretrained.WEIGHTS.values()}


def test_model_card_rewrites_local_links_and_keeps_one_disclosure():
    root = Path(__file__).resolve().parents[1]
    card = model_card((root / "README.md").read_text())
    assert "Release staging:" not in card
    assert "(weights/" not in card and "(tests/results/" not in card and "(docs/" not in card
    assert "https://github.com/sevagh/gaborformer/blob/main/docs/results.md#a-trained-on-musdb18-hq-only" in card
    assert "https://github.com/sevagh/gaborformer/blob/main/weights/README.md" in card
    assert "license_link: LICENSE" in card
    assert "https://github.com/sevagh/gaborformer/blob/main/LICENSE" in card
    assert card.count("## AI assistance") == 1


def test_card_only_cli_needs_no_third_party_packages(tmp_path):
    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "model-card"
    # -S excludes site-packages, including torch and huggingface_hub.
    subprocess.run(
        [sys.executable, "-S", "-m", "scripts.prepare_huggingface",
         "--card-only", "--out-dir", str(destination)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert sorted(path.name for path in destination.iterdir()) == ["README.md"]
    assert (destination / "README.md").read_text() == model_card((root / "README.md").read_text())


def test_cli_rejects_colliding_output_names_before_downloading(monkeypatch, capsys):
    import importlib
    separate = importlib.import_module("gaborformer.separate")

    def unexpected_download(*args, **kwargs):
        pytest.fail("colliding output names reached checkpoint download")

    monkeypatch.setattr(separate, "download_weights", unexpected_download)
    with pytest.raises(SystemExit) as exc:
        separate.main(["a/song.wav", "b/song.flac"])
    assert exc.value.code == 2
    assert "different --out paths" in capsys.readouterr().err
