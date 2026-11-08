# CLI usage

[← README](../README.md)

## Installation and inference

Requires Python 3.10+ and PyTorch 2.4+. Install a CUDA-enabled PyTorch build for GPU inference; CPU inference is also supported. From a clone of this repository:

```bash
python -m pip install -e .

gaborformer song.wav --out separated
```

If you use [uv](https://docs.astral.sh/uv/), the same `pyproject.toml` works with `uv run gaborformer song.wav --out separated`; uv creates the environment and installs dependencies automatically. The pip workflow above remains fully supported.

The first run downloads the 533 MB FP16 checkpoint from [Hugging Face][hf] and verifies its SHA-256 checksum. Later runs reuse the local cache. Public checkpoints require no Hugging Face account.

## Checkpoints and offline use

See [pretrained weights](../weights/README.md) for all three checkpoints, R2 mirror downloads, and checksums. To use a checkpoint you already have:

```bash
gaborformer song.wav --weights weights/gaborformer-fp16.safetensors --out separated
```

Use `--offline` to require cached or local weights without contacting Hugging Face.

For private checkpoints, authenticate with `hf auth login` using an account with access, or pass a local checkpoint with `--weights`.

## Audio formats and output

The equivalent module command is `python -m gaborformer.separate`. **M4A/AAC input is supported through FFmpeg**, included in the Docker image; for a local installation, install `ffmpeg` on your system. Mono and stereo inputs at other sample rates are **automatically resampled to 44.1 kHz** using [SoXR](https://python-soxr.readthedocs.io/) in high-quality mode. Mono is duplicated to stereo. Output goes to `separated/song/{vocals,drums,bass,other}.wav` as 44.1 kHz stereo, 32-bit float audio.

## CLI options

Multiple input files are supported. Use `--batch-size 1` to reduce memory use, `--cpu` to select CPU inference, or `--precision fp16` for CUDA GPUs without BF16 support.

Run `gaborformer --help` for the complete option list. For example:

```bash
gaborformer first.wav second.m4a --out separated --batch-size 1
gaborformer song.wav --cpu --out separated
```

## Compilation and measured speed

`--compile` is opt-in and compiles the transformer layers. It is most useful for long tracks or multiple files in one invocation:

```bash
gaborformer first.wav second.wav --compile --out separated
```

On an idle RTX 3090, a 210-second track took the following **end-to-end CLI times**, including process startup, checkpoint loading, and writing four stems:

| Mode | Wall time |
|---|---:|
| Default, without compilation | 20.8 s |
| Compiled, cold disk cache | 28.1 s |
| Compiled, warm disk cache | 19.4 s |

Steady-state inference took **16.2 s without compilation** and **11.5 s compiled** (1.41× faster), excluding startup, weight loading, and output writing. Cached runs still have setup overhead: the end-to-end saving on this single track was 1.4 s. Longer tracks and multiple files amortize that overhead. All measurements used BF16 computation, 50% overlap, and batch size 8. See [full-test-set validation](results.md#released-checkpoint-validation) for the precision and compilation results.

Compiled kernels persist under `~/.cache/gaborformer/inductor` by default, including across reboots. Setting `XDG_CACHE_HOME` relocates the cache to `$XDG_CACHE_HOME/gaborformer/inductor`; an explicit `TORCHINDUCTOR_CACHE_DIR` takes precedence. Reuse requires a compatible PyTorch and hardware configuration.

[hf]: https://huggingface.co/sevagh/gaborformer
