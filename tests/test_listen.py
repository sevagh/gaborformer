"""CPU checks for comparison output integrity and process isolation."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import soundfile as sf

from scripts import listen
from scripts.competitors import sw_four_stems


def test_resample_and_fold_without_gain_change(tmp_path):
    path = tmp_path / "mono 48k.wav"
    sf.write(path, np.full(4800, 0.25, dtype=np.float32), 48000, subtype="FLOAT")
    audio = listen.read_audio(path)
    assert audio.shape == (4410, 2)
    np.testing.assert_array_equal(audio[:, 0], audio[:, 1])
    np.testing.assert_allclose(audio[100:-100], 0.25, atol=1e-5)
    native = {name: np.full((2, 20), i, dtype=np.float32)
              for i, name in enumerate(("vocals", "drums", "bass", "other", "guitar", "piano"))}
    stems = sw_four_stems(native)
    assert set(stems) == set(listen.STEMS)
    np.testing.assert_array_equal(stems["other"], 12)
    listen.write_stems(tmp_path / "stems", stems, 20)
    output, _ = sf.read(tmp_path / "stems/other.wav")
    np.testing.assert_array_equal(output, 12)  # FLOAT output must not clip at 1.


def test_comparison_processes_and_failed_rerun(tmp_path, monkeypatch):
    source = tmp_path / "a song.wav"
    sf.write(source, np.zeros((64, 2), dtype=np.float32), 44100, subtype="FLOAT")
    out = tmp_path / "new" / "nested" / "separated"
    assert not out.exists()
    seen = []

    def fake_worker(command, check):
        name = command[command.index("--worker") + 1]
        seen.append(name)
        destination = Path(command[command.index("--out") + 1])
        listen.write_stems(destination, {stem: np.zeros((2, 64)) for stem in listen.STEMS}, 64)
        (destination / "model.json").write_text("{}")

    monkeypatch.setattr(subprocess, "run", fake_worker)
    arguments = [str(source), "--out", str(out), "--include-competitors", "--cpu"]
    listen.main(arguments)
    assert seen == list(listen.MODELS)
    destination = out / "a song"
    assert len(list(destination.glob("*/*.wav"))) == 12
    assert json.loads((destination / "comparison.json").read_text())["status"] == "complete"
    original = (destination / "gaborformer/vocals.wav").read_bytes()

    def fail_worker(command, check):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail_worker)
    with pytest.raises(SystemExit, match="1"):
        listen.main(arguments)
    assert json.loads((destination / "comparison.json").read_text())["status"] == "failed"
    assert (destination / "gaborformer/vocals.wav").read_bytes() == original

    sf.write(source, np.ones((64, 2), dtype=np.float32), 44100, subtype="FLOAT")
    with pytest.raises(SystemExit, match="1"):
        listen.main(arguments)
    assert (destination / "gaborformer/vocals.wav").read_bytes() == original


def test_reject_invalid_stems_before_writing(tmp_path):
    stems = {name: np.zeros((2, 10)) for name in listen.STEMS}
    stems["other"][0, 0] = np.nan
    with pytest.raises(RuntimeError, match="Invalid other"):
        listen.write_stems(tmp_path / "stems", stems, 10)
    assert not (tmp_path / "stems").exists()
