"""Deterministic reference clip and stem checksums, for anyone porting the model (ONNX, C++, other runtimes).

    python scripts/reference_output.py <weights> --write reference/    # make the reference
    python scripts/reference_output.py <weights> --check reference/    # compare a runtime

The clip is synthesised here (no MUSDB audio is redistributed): 12 s of stereo audio at 44.1 kHz made of a seeded
noise floor, a kick-like exponentially decaying pulse train, a bass sine with vibrato, a chord of harmonics and a
chirp, at fixed gains. `--write` separates it at fp32 with overlap 0.5 and batch 1 (the reference numerics: no
autocast, no chunk-batch effects) and stores clip.wav, stems.npz (float32, stems x channels x samples in
vocals/drums/bass/other order) and sha256.txt. `--check` reruns and reports max-abs and SDR against the stored
stems; a correct port at fp32 matches to ~1e-4 max-abs, a bf16/fp16 runtime to ~1e-2 with SDR above 40 dB.
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

from gaborformer.model import STEMS
from gaborformer.separate import load_model, separate

SR = 44100


def make_clip(seconds=12.0, seed=1234):
    g = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    noise = 0.01 * g.standard_normal((2, n)).astype(np.float32)
    kick = np.zeros(n, np.float32)
    for k in range(int(seconds * 2)):  # 120 bpm pulse train
        i = int(k * 0.5 * SR)
        env = np.exp(-np.arange(min(8000, n - i)) / 1500.0)
        kick[i:i + len(env)] += (0.6 * np.sin(2 * np.pi * (55 + 40 * np.exp(-np.arange(len(env)) / 400.0)) * np.arange(len(env)) / SR) * env).astype(np.float32)
    bass = (0.25 * np.sin(2 * np.pi * 61.7 * t + 0.3 * np.sin(2 * np.pi * 5.0 * t))).astype(np.float32)
    chord = sum(0.08 / (h + 1) * np.sin(2 * np.pi * f * h * t) for f in (261.6, 329.6, 392.0) for h in range(1, 6)).astype(np.float32)
    chirp = (0.1 * np.sin(2 * np.pi * (300 * t + 400 * t ** 2 / seconds))).astype(np.float32)
    left = noise[0] + kick + bass + 0.9 * chord + 0.6 * chirp
    right = noise[1] + kick + bass + 0.7 * chord + 1.0 * chirp
    clip = np.stack([left, right]).astype(np.float32)
    return clip / max(1.0, np.abs(clip).max() / 0.9)


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("weights")
    p.add_argument("--write", help="directory to create the reference in")
    p.add_argument("--check", help="directory holding a reference to compare against")
    p.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args(argv)
    if bool(a.write) == bool(a.check):
        raise SystemExit("give exactly one of --write or --check")
    device = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    model, ck = load_model(a.weights, device)
    clip = make_clip()
    stems = separate(model, torch.from_numpy(clip).to(device), batch_size=a.batch_size, overlap=a.overlap, precision=a.precision).cpu().numpy()
    if a.write:
        import soundfile as sf
        os.makedirs(a.write, exist_ok=True)
        sf.write(os.path.join(a.write, "clip.wav"), clip.T, SR, subtype="FLOAT")
        np.savez(os.path.join(a.write, "stems.npz"), **{s: stems[i] for i, s in enumerate(STEMS)}, clip=clip)
        info = {"checkpoint": os.path.basename(a.weights), "step": ck.get("step"), "cfg": ck["cfg"], "precision": a.precision, "overlap": a.overlap,
                "batch_size": a.batch_size, "device": device, "torch": torch.__version__, "clip_sha256": digest(clip),
                "stem_sha256": {s: digest(stems[i]) for i, s in enumerate(STEMS)}}
        json.dump(info, open(os.path.join(a.write, "reference.json"), "w"), indent=1)
        with open(os.path.join(a.write, "sha256.txt"), "w") as f:
            f.write(f"{info['clip_sha256']}  clip (float32 2x{clip.shape[1]})\n")
            for s in STEMS:
                f.write(f"{info['stem_sha256'][s]}  {s} (float32 2x{clip.shape[1]})\n")
        print(json.dumps({k: v for k, v in info.items() if k != "cfg"}, indent=1))
    else:
        ref = np.load(os.path.join(a.check, "stems.npz"))
        info = json.load(open(os.path.join(a.check, "reference.json")))
        assert digest(ref["clip"]) == info["clip_sha256"] == digest(clip), "the synthesised clip differs from the reference clip"
        ok = True
        for i, s in enumerate(STEMS):
            r, e = ref[s], stems[i]
            maxabs = float(np.abs(r - e).max())
            sdr = float(10 * np.log10((r.astype(np.float64) ** 2).sum() / max(((r - e).astype(np.float64) ** 2).sum(), 1e-20)))
            exact = digest(e) == info["stem_sha256"][s]
            ok &= sdr > 40
            print(f"{s:7s} max|diff| {maxabs:.2e}  SDR vs reference {sdr:6.1f} dB  bit-exact {exact}")
        print("PASS" if ok else "FAIL: a stem is more than 40 dB from the reference")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
