"""Full-track separation by overlap-adding fixed-length chunks, and loading released weights.

    python -m gaborformer.separate song.wav [more.wav ...] [--weights /path/to/checkpoint] [--out separated]
        [--precision bf16|fp16|fp32] [--overlap 0.5|0.75] [--compile] [--batch-size 8] [--cpu]

writes <out>/<song>/{vocals,drums,bass,other}.wav at 44.1 kHz. Input is automatically resampled with SoXR when
needed; mono is duplicated to stereo. The released FP16 weights are downloaded from Hugging Face and cached.
--weights accepts a local .safetensors file (fp32 or fp16 weights; fp16 is upcast to fp32 at load) or a training
checkpoint (.pt). --offline requires cached weights or an explicit local file.

Cost of the options, in forward passes per second of audio relative to the default (bf16 autocast, overlap 0.5):
  --precision fp32   same passes, without CUDA autocast; the reference numerics
  --precision fp16   same passes as bf16; for GPUs without bf16 (pre-Ampere)
  --overlap 0.75     approximately 2x the passes (hop is a quarter chunk instead of half)
  --compile          same passes, ~1.4x faster forward passes on an RTX 3090; first-run compilation adds ~7 s on the
                     measured track. The disk cache persists across runs; best for long files or several files
"""
import argparse
import json
import os

import torch

from .model import STEMS, Gaborformer
from .pretrained import download_weights

PRECISIONS = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}


@torch.no_grad()
def separate(model, mix, batch_size=8, overlap=0.5, precision="bf16"):
    """mix: (channels, n) -> stems (n_stems, channels, n).

    Chunks of model.length are crossfaded with triangular windows. The NSGT is circular over a chunk, so chunk edges
    get the least weight. `overlap` is the fraction of a chunk shared with its neighbour (0.5: each sample is covered
    by 2 chunks; 0.75: by 4, at twice the forward passes). `precision` selects the CUDA autocast dtype for the forward
    pass; on the CPU the model runs in fp32.
    """
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if mix.ndim != 2 or mix.shape[-1] == 0:
        raise ValueError("mix must be a nonempty (channels, samples) tensor")
    L = model.length
    hop = max(1, int(round(L * (1 - overlap))))
    pad = L - hop
    n = mix.shape[-1]
    total = pad + n + pad
    total += (-(total - L)) % hop
    x = torch.nn.functional.pad(mix, (pad, total - n - pad))
    starts = list(range(0, total - L + 1, hop))
    win = 1 - torch.linspace(-1, 1, L + 2, device=mix.device)[1:-1].abs()
    out = torch.zeros(model.n_stems, mix.shape[0], total, device=mix.device)
    norm = torch.zeros(total, device=mix.device)
    dtype = PRECISIONS[precision]
    for i in range(0, len(starts), batch_size):
        s = starts[i:i + batch_size]
        chunk = torch.stack([x[:, a:a + L] for a in s])
        with torch.autocast("cuda", dtype=dtype or torch.bfloat16, enabled=mix.is_cuda and dtype is not None):
            y = model(chunk).float()
        for a, ya in zip(s, y):
            out[..., a:a + L] += ya * win
            norm[a:a + L] += win
    return (out / norm.clamp_min(1e-8))[..., pad:pad + n]


def usdr(ref, est):
    """Utterance-level SDR (MDX 'new SDR'): energy ratio over the whole track, per stem. ref/est: (..., ch, n)."""
    num = ref.double().pow(2).sum((-2, -1))
    den = (ref.double() - est.double()).pow(2).sum((-2, -1))
    return 10 * torch.log10((num + 1e-8) / (den + 1e-8))


def read_checkpoint(path):
    """.safetensors (weights + cfg metadata) or a training .pt -> dict with cfg, model (state dict), step, val."""
    path = os.fspath(path)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file, safe_open
        with safe_open(path, framework="pt") as f:
            meta = f.metadata() or {}
        return {"cfg": json.loads(meta["cfg"]), "step": json.loads(meta.get("step", "null")), "val": json.loads(meta.get("val", "null")),
                "model": load_file(path), "source_sha256": meta.get("source_sha256"), "weights_dtype": meta.get("dtype")}
    return torch.load(path, map_location="cpu", weights_only=True)


def compile_cache_dir():
    """Persistent kernel cache: explicit PyTorch override, then XDG_CACHE_HOME, then ~/.cache."""
    if "TORCHINDUCTOR_CACHE_DIR" in os.environ:
        return os.environ["TORCHINDUCTOR_CACHE_DIR"]
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "gaborformer", "inductor")


def load_model(path, device="cuda", compile=False):
    """-> (model in eval mode on device, checkpoint dict). Weights load into an fp32 model whatever the file's dtype;
    `compile=True` torch.compiles the transformers (~1.4x faster forward passes on an RTX 3090). Generated kernels
    persist in `compile_cache_dir()` for reuse by later runs with a compatible PyTorch and hardware configuration."""
    ck = read_checkpoint(path)
    model = Gaborformer.from_config(ck["cfg"])
    model.load_state_dict(ck["model"])
    if compile:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = compile_cache_dir()
        model.compile_transformers()
    return model.to(device).eval(), ck


def main(argv=None):
    import soundfile as sf
    from .audio import read_audio

    p = argparse.ArgumentParser(description="Separate audio files into vocals, drums, bass and other.")
    p.add_argument("audio", nargs="+", help="mono or stereo audio; automatically resampled to 44.1 kHz")
    p.add_argument("--weights", help="local .safetensors or .pt checkpoint; default: download the released FP16 weights")
    p.add_argument("--offline", action="store_true", help="use only cached or local weights, without contacting Hugging Face")
    p.add_argument("--revision", help="Hugging Face revision to download; primarily for testing a private release")
    p.add_argument("--out", default="separated")
    p.add_argument("--precision", choices=sorted(PRECISIONS), default="bf16", help="CUDA autocast dtype for the forward pass (default bf16)")
    p.add_argument("--overlap", type=float, default=0.5, help="fraction of each chunk shared with its neighbour; 0.75 approximately doubles the forward passes")
    p.add_argument("--compile", action="store_true", help="torch.compile the transformers with a persistent disk cache; best for long files or multiple files")
    p.add_argument("--batch-size", type=int, default=8, help="chunks per forward pass; lower it if the GPU runs out of memory")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args(argv)
    names = [os.path.normcase(os.path.splitext(os.path.basename(path))[0]) for path in a.audio]
    if len(set(names)) != len(names):
        p.error("input filenames would share an output directory; run them separately with different --out paths")
    if a.weights and a.revision:
        p.error("--revision applies only to Hugging Face downloads; omit it when using --weights")
    if a.weights:
        if not os.path.isfile(a.weights):
            p.error(f"checkpoint not found: {a.weights}")
        weights = a.weights
    else:
        try:
            weights = download_weights(offline=a.offline, revision=a.revision)
        except RuntimeError as exc:
            p.error(str(exc))
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    model, _ = load_model(weights, device, compile=a.compile)
    for path in a.audio:
        try:
            audio = read_audio(path)
        except (OSError, ValueError, RuntimeError) as exc:
            p.error(str(exc))
        mix = torch.from_numpy(audio.T.copy())
        stems = separate(model, mix.to(device), batch_size=a.batch_size, overlap=a.overlap, precision=a.precision).cpu()
        d = os.path.join(a.out, os.path.splitext(os.path.basename(path))[0])
        os.makedirs(d, exist_ok=True)
        for name, y in zip(STEMS, stems):
            sf.write(os.path.join(d, name + ".wav"), y.T.numpy(), 44100, subtype="FLOAT")
        print(f"{path} -> {d}/", flush=True)


if __name__ == "__main__":
    main()
