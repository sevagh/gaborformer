"""Partial evaluations must never be relabeled as a different benchmark run."""
import json

import pytest

from scripts import evaluate


@pytest.fixture
def partial_run(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"checkpoint one")
    root = tmp_path / "musdb"
    (root / "train").mkdir(parents=True)
    tracks = []
    for name in ("Track A", "Track B"):
        track = root / "test" / name
        track.mkdir(parents=True)
        for stem in ("mixture", *evaluate.STEMS):
            (track / (stem + ".wav")).write_bytes((name + stem).encode())
        tracks.append(str(track))
    inference = {"precision": "bf16", "overlap": 0.5, "compile": False, "batch_size": 8}
    out = tmp_path / "results.json"
    partial = str(out) + ".partial.jsonl"
    identity = evaluate.evaluation_identity(weights, tracks, inference)
    assert evaluate.resume_scores(partial, identity) == {}
    score = {"track": "Track A", "csdr": [1, 2, 3, 4], "usdr": [5, 6, 7, 8]}
    with open(partial, "a") as f:
        f.write(json.dumps(score) + "\n")
    return weights, root, tracks, inference, out, partial, score


def test_same_evaluation_resumes_completed_scores(partial_run):
    weights, _, tracks, inference, _, partial, score = partial_run
    identity = evaluate.evaluation_identity(weights, tracks, inference)
    assert evaluate.resume_scores(partial, identity) == {
        score["track"]: {key: score[key] for key in ("csdr", "usdr")}
    }


@pytest.mark.parametrize("change", ["checkpoint", "audio", "subset", "precision", "overlap", "compile", "batch_size"])
def test_changed_evaluation_is_rejected_before_model_load(partial_run, monkeypatch, change):
    weights, root, tracks, _, out, partial, _ = partial_run
    before = open(partial).read()
    args = [str(weights), "--musdb", str(root), "--out", str(out)]
    if change == "checkpoint":
        weights.write_bytes(b"checkpoint two")
    elif change == "audio":
        (root / "test" / "Track A" / "vocals.wav").write_bytes(b"changed audio")
    else:
        args += {
            "subset": ["--tracks", "1"],
            "precision": ["--precision", "fp16"],
            "overlap": ["--overlap", "0.75"],
            "compile": ["--compile"],
            "batch_size": ["--batch-size", "1"],
        }[change]

    def no_model(*args, **kwargs):
        pytest.fail("mismatched evaluation reached model loading")

    monkeypatch.setattr(evaluate, "load_model", no_model)
    with pytest.raises(SystemExit) as exc:
        evaluate.main(args)
    assert exc.value.code == 2
    assert not out.exists()
    assert open(partial).read() == before


@pytest.mark.parametrize("kind", ["legacy", "duplicate", "unexpected", "truncated"])
def test_unidentifiable_or_invalid_partial_scores_are_rejected(partial_run, kind):
    weights, _, tracks, inference, _, partial, score = partial_run
    identity = evaluate.evaluation_identity(weights, tracks, inference)
    if kind == "legacy":
        with open(partial, "w") as f:
            f.write(json.dumps(score) + "\n")
    else:
        with open(partial, "a") as f:
            if kind == "unexpected":
                score = {**score, "track": "Not selected"}
            f.write('{"track":' if kind == "truncated" else json.dumps(score) + "\n")
    with pytest.raises(ValueError, match="Use a new --out"):
        evaluate.resume_scores(partial, identity)
