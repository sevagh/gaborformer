"""Decode local audio and prepare 44.1 kHz stereo input for inference."""
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import soxr


def _read_with_ffmpeg(path):
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError(
            f"Cannot decode {path.name} with SoundFile. Install FFmpeg for M4A/AAC "
            "and other compressed formats, or use the Docker image, which includes it."
        )
    # Preserve the source sample rate and channels; SoXR performs any resampling below.
    # An argument list and absolute input path keep spaces and shell characters literal.
    with tempfile.TemporaryDirectory(prefix="gaborformer-audio-") as directory:
        decoded = Path(directory) / "decoded.wav"
        result = subprocess.run(
            [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-threads", "1",
             "-i", str(path), "-map", "0:a:0", "-vn", "-sn", "-dn",
             "-c:a", "pcm_f32le", "-threads", "1", "-f", "wav", "-n", str(decoded)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        if result.returncode:
            raise RuntimeError(f"FFmpeg could not decode {path.name}: {result.stderr.strip()[-2000:]}")
        return sf.read(decoded, dtype="float32", always_2d=True)


def read_audio(path):
    """Return finite stereo float32 samples, shape (samples, 2), at 44.1 kHz.

    SoundFile handles supported formats directly. FFmpeg is the fallback for
    formats such as M4A/AAC; it decodes without changing gain or channel layout.
    """
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Expected an audio file: {path}")
    try:
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
    except RuntimeError:
        audio, rate = _read_with_ffmpeg(path)
    if not len(audio) or audio.shape[1] not in (1, 2) or not np.isfinite(audio).all():
        raise ValueError(f"{path.name}: input must contain finite, nonempty mono or stereo audio")
    if rate != 44100:
        audio = soxr.resample(audio, rate, 44100, quality="HQ")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio
