"""Exercise file loading, resampling, and stem writing without loading the neural network."""
import importlib
import shutil
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import soxr
import torch

from gaborformer.model import STEMS
from gaborformer.audio import read_audio

inference = importlib.import_module("gaborformer.separate")


class Repeat(torch.nn.Module):
    length, n_stems = 4410, 4

    def forward(self, chunk):
        return chunk[:, None].expand(-1, self.n_stems, -1, -1)


@pytest.mark.parametrize("sample_rate,channels", [(22050, 1), (44100, 2), (48000, 2), (96000, 2)])
def test_cli_preserves_pitch_duration_and_channels(tmp_path, monkeypatch, sample_rate, channels):
    t = np.arange(sample_rate // 5) / sample_rate
    audio = np.stack([0.25 * np.sin(2 * np.pi * 440 * t), 0.125 * np.sin(2 * np.pi * 880 * t)], axis=1)
    path = tmp_path / "song.wav"
    sf.write(path, audio[:, :channels], sample_rate, subtype="FLOAT")
    out = tmp_path / "stems"
    checkpoint = tmp_path / "local.pt"
    checkpoint.touch()
    monkeypatch.setattr(inference, "load_model", lambda *args, **kwargs: (Repeat(), {}))
    def unexpected_download(*args, **kwargs):
        pytest.fail("local weights should bypass Hugging Face")
    monkeypatch.setattr(inference, "download_weights", unexpected_download)
    monkeypatch.setattr(sys, "argv", ["gaborformer", str(path), "--weights", str(checkpoint), "--offline", "--out", str(out), "--cpu"])
    if sample_rate == 44100:
        def unexpected_resample(*args, **kwargs):
            pytest.fail("44.1 kHz audio should bypass resampling")
        monkeypatch.setattr(soxr, "resample", unexpected_resample)

    inference.main()

    t_out = np.arange(44100 // 5) / 44100
    expected = np.stack([0.25 * np.sin(2 * np.pi * 440 * t_out), 0.125 * np.sin(2 * np.pi * 880 * t_out)], axis=1)
    if channels == 1:
        expected[:, 1] = expected[:, 0]
    for stem in STEMS:
        result, rate = sf.read(out / "song" / f"{stem}.wav", dtype="float32", always_2d=True)
        assert rate == 44100
        assert result.shape == expected.shape
        # Ignore filter boundary transients; check against the original tones at the output time grid.
        np.testing.assert_allclose(result[256:-256], expected[256:-256], atol=2e-5)
        assert sf.info(out / "song" / f"{stem}.wav").subtype == "FLOAT"


@pytest.mark.parametrize("offline", [False, True])
def test_cli_resolves_default_checkpoint(tmp_path, monkeypatch, offline):
    path = tmp_path / "song.wav"
    sf.write(path, np.zeros((441, 2), dtype="float32"), 44100)
    checkpoint = tmp_path / "cached.safetensors"
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return checkpoint

    def load(weights, device, compile):
        assert weights == checkpoint and device == "cpu"
        return Repeat(), {}

    monkeypatch.setattr(inference, "download_weights", download)
    monkeypatch.setattr(inference, "load_model", load)
    argv = ["gaborformer", str(path), "--out", str(tmp_path / "stems"), "--cpu"]
    if offline:
        argv.append("--offline")
    monkeypatch.setattr(sys, "argv", argv)
    inference.main()
    assert calls == [{"offline": offline, "revision": None}]
    assert len(list((tmp_path / "stems/song").glob("*.wav"))) == 4


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is needed to encode the AAC fixture")
@pytest.mark.parametrize("channels", [1, 2])
def test_m4a_decoder_and_cli(tmp_path, monkeypatch, channels):
    from scripts import listen

    t = np.arange(48000) / 48000
    audio = np.stack([0.25 * np.sin(2 * np.pi * 440 * t),
                      0.125 * np.sin(2 * np.pi * 880 * t)], axis=1)
    wav = tmp_path / "source.wav"
    sf.write(wav, audio[:, :channels], 48000, subtype="FLOAT")
    path = tmp_path / "sample [brackets] $literal.m4a"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(wav),
                    "-c:a", "aac", "-b:a", "192k", "-threads", "1", str(path)], check=True)

    decoded = read_audio(path)
    np.testing.assert_array_equal(listen.read_audio(path), decoded)
    assert decoded.shape[1] == 2
    # AAC may pad the final frame; account for that without trimming real input.
    assert 44100 <= len(decoded) < 44100 + 1024
    t_out = np.arange(44100) / 44100
    expected = np.stack([0.25 * np.sin(2 * np.pi * 440 * t_out),
                         0.125 * np.sin(2 * np.pi * 880 * t_out)], axis=1)
    if channels == 1:
        expected[:, 1] = expected[:, 0]
        np.testing.assert_array_equal(decoded[:, 0], decoded[:, 1])
    np.testing.assert_allclose(decoded[2048:42000], expected[2048:42000], atol=0.025)

    checkpoint = tmp_path / "local.pt"
    checkpoint.touch()
    out = tmp_path / "new" / "nested" / "stems"
    monkeypatch.setattr(inference, "load_model", lambda *args, **kwargs: (Repeat(), {}))
    monkeypatch.setattr(sys, "argv", ["gaborformer", str(path), "--weights", str(checkpoint),
                                     "--out", str(out), "--offline", "--cpu"])
    inference.main()
    for stem in STEMS:
        result, rate = sf.read(out / path.stem / f"{stem}.wav", dtype="float32")
        assert rate == 44100
        np.testing.assert_allclose(result, decoded, atol=1e-6)


def test_decoder_reports_missing_ffmpeg_and_preserves_wav_support(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    path = tmp_path / "song.wav"
    sf.write(path, np.zeros((441, 2), dtype=np.float32), 44100)
    assert read_audio(path).shape == (441, 2)
    path = tmp_path / "unsupported.m4a"
    path.write_bytes(b"not a SoundFile format")
    with pytest.raises(RuntimeError, match="Install FFmpeg"):
        read_audio(path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_decoder_rejects_corrupt_input(tmp_path):
    path = tmp_path / "corrupt.m4a"
    path.write_bytes(b"not an audio file")
    with pytest.raises(RuntimeError, match="FFmpeg could not decode"):
        read_audio(path)
