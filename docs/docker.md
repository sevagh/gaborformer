# Docker and listening comparisons

[← README](../README.md)

The image includes FFmpeg, CUDA 12.4 and cuDNN. GPU inference requires the NVIDIA Container Toolkit. Build from the repository root, then run a three-model listening comparison on your own track:

```bash
docker build -t gaborformer .

TRACK="/absolute/path/to/your-track.m4a"

docker run --rm --gpus all \
  --device=/dev/nvidia0 \
  --device=/dev/nvidiactl \
  --device=/dev/nvidia-uvm \
  --user "$(id -u):$(id -g)" \
  -v "$(dirname "$TRACK"):/input:ro" \
  -v "$PWD:/work" \
  -v gaborformer-cache:/cache \
  gaborformer "/input/$(basename "$TRACK")" \
  --out /work/separated --include-competitors
```

The runner creates `separated/` and its subdirectories as your user; no manual `mkdir` is needed. Existing output from a different input is rejected. Remove `--include-competitors` to run Gaborformer alone. With it, [DiCoSe](https://github.com/Russell-Izadi-Bose/DiCoSe) and [BS-RoFormer-SW](https://github.com/openmirlab/bs-roformer-infer) run sequentially after Gaborformer. Each folder contains `vocals.wav`, `drums.wav`, `bass.wav`, and `other.wav`, plus checkpoint hashes and settings:

```text
separated/<track>/
├── gaborformer/
├── dicose/
└── bs-roformer-sw/
```

All models receive the same resampled 44.1 kHz stereo input, with no additional loudness normalization. SW's guitar and piano outputs are added to “other.” **SW is not the SDX'23-winning checkpoint in [leaderboard B](results.md#b-systems-trained-with-extra-data).** DiCoSe uses the authors' one-step consistency refiner with 11-second chunks and 50% overlap; this listening runner does not reproduce the leaderboard evaluation protocol.

## Downloading and caching weights

Weights download on first use and persist in `gaborformer-cache` (about **5.9 GB** for all three systems). Add `--offline` to use only cached weights. To download ahead of time without using the GPU:

```bash
docker run --rm --user "$(id -u):$(id -g)" -v gaborformer-cache:/cache \
  gaborformer --include-competitors --download-only
```

## GPU access and private checkpoints

The explicit `--device` mappings handle NVML errors on some Linux hosts. For CPU inference, omit `--gpus all` and all three `--device` options and add `--cpu`.

For private Hugging Face checkpoints, add these Docker options to use your existing login: `-v "$HOME/.cache/huggingface/token:/run/secrets/hf_token:ro" -e HF_TOKEN_PATH=/run/secrets/hf_token`. Alternatively, mount a local checkpoint and use `--weights`. Competitor code and weights retain their upstream licenses and terms.
