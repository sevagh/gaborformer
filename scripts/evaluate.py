"""Evaluate weights on the MUSDB18-HQ test set: museval SDR (SiSEC 2018 protocol) and uSDR.

Separation runs on the GPU, museval (BSSEval v4, 1 s windows) on a pool of CPU workers (each in-flight full-length
track costs ~6 GB of RAM). Reported, per stem and averaged over stems:
  cSDR: median over tracks of each track's median over 1 s windows (the number papers quote),
  uSDR: mean over tracks of whole-track SDR (MDX 'new SDR', what training validates on).
Per-track scores go to --out (JSON) for paired comparisons; each finished track is appended to <out>.partial.jsonl
and a rerun skips the tracks already there only when checkpoint contents, input audio, and inference
settings match. Use a different --out for a different evaluation.

    python scripts/evaluate.py weights/gaborformer-fp16.safetensors --musdb /data/musdb18hq --workers 4 --out results.json
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from gaborformer.data import list_tracks, load_track, musdb_root
from gaborformer.model import STEMS
from gaborformer.pretrained import sha256
from gaborformer.separate import load_model, separate, usdr

WINDOW = HOP = 44100


def bss_eval(refs, ests):
    """refs, ests: (stems, channels, n) float32 numpy -> per-stem median SDR over 1 s windows."""
    import museval
    sdr, _, _, _ = museval.evaluate(refs.transpose(0, 2, 1), ests.transpose(0, 2, 1), win=WINDOW, hop=HOP)
    return [float(np.nanmedian(s)) for s in sdr]


def evaluation_identity(weights, tracks, inference):
    """Bind resumable scores to checkpoint bytes, selected audio, and inference settings."""
    return {
        "version": 1,
        "checkpoint_sha256": sha256(weights),
        "inference": inference,
        "metric": {"name": "BSSEval v4", "museval_version": version("museval"),
                   "window_samples": WINDOW, "hop_samples": HOP,
                   "aggregation": "median windows, median tracks, mean stems"},
        "tracks": {
            Path(path).name: {
                name: sha256(Path(path) / (name + ".wav"))
                for name in ("mixture", *STEMS)
            }
            for path in tracks
        },
    }


def resume_scores(partial, identity):
    """Reject stale or unidentifiable partial results instead of silently relabeling them."""
    path = Path(partial)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as f:
            f.write(json.dumps({"evaluation": identity}) + "\n")
        return {}
    try:
        with path.open() as f:
            if json.loads(next(f)) != {"evaluation": identity}:
                raise ValueError("checkpoint, input audio, or inference settings differ (or the file has no identity header)")
            scores = {}
            for line in f:
                record = json.loads(line)
                name = record.pop("track")
                if name not in identity["tracks"] or name in scores:
                    raise ValueError("unexpected or duplicate track")
                if set(record) != {"csdr", "usdr"} or any(
                    not isinstance(record[key], list) or len(record[key]) != len(STEMS)
                    or any(not isinstance(value, (int, float)) for value in record[key])
                    for key in ("csdr", "usdr")
                ):
                    raise ValueError("invalid score record")
                scores[name] = record
        return scores
    except (ValueError, KeyError, TypeError, StopIteration, AttributeError) as exc:
        raise ValueError(
            f"Cannot resume {path}: {exc}. Use a new --out path, or remove the partial file to restart."
        ) from exc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("weights")
    p.add_argument("--musdb", help="MUSDB18-HQ root; default $MUSDB_PATH")
    p.add_argument("--tracks", type=int, default=50)
    p.add_argument("--workers", type=int, default=4, help="museval processes (~6 GB RAM each on full-length tracks)")
    p.add_argument("--batch-size", type=int, default=8, help="chunks per forward pass; lower it if the GPU runs out of memory")
    p.add_argument("--out")
    p.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="bf16", help="CUDA autocast dtype for the forward pass (default bf16, the published setting)")
    p.add_argument("--overlap", type=float, default=0.5, help="chunk overlap fraction (0.5 default; 0.75 doubles the forward passes)")
    p.add_argument("--compile", action="store_true", help="torch.compile the transformers before evaluating")
    a = p.parse_args(argv)
    if min(a.tracks, a.workers, a.batch_size) < 1:
        p.error("--tracks, --workers, and --batch-size must be positive")
    tracks = list_tracks(musdb_root(a.musdb), "test")[:a.tracks]
    if not tracks:
        p.error("no test tracks found")
    inference = {"precision": a.precision, "overlap": a.overlap, "compile": a.compile, "batch_size": a.batch_size}
    partial = a.out + ".partial.jsonl" if a.out else None
    identity = None
    res, pending = {}, {}
    if partial:
        identity = evaluation_identity(a.weights, tracks, inference)
        try:
            res = resume_scores(partial, identity)
        except ValueError as exc:
            p.error(str(exc))
        if res:
            print(f"resuming: {len(res)} tracks already scored in {partial}", flush=True)

    model, ck = load_model(a.weights, compile=a.compile)
    print(f"{ck['cfg']['frontend']} / {ck['cfg']['bands']} bands, step {ck.get('step')}", flush=True)
    print(f"inference: precision {a.precision}, overlap {a.overlap}, compile {a.compile}, batch {a.batch_size}", flush=True)

    def collect(name):
        res[name]["csdr"] = pending.pop(name).result()
        if partial:
            with open(partial, "a") as f:
                f.write(json.dumps({"track": name, **res[name]}) + "\n")
        print(f"  [{sum('csdr' in r for r in res.values())}/{len(tracks)}] {name}: cSDR " + " ".join(f"{v:5.2f}" for v in res[name]["csdr"]), flush=True)

    with ProcessPoolExecutor(a.workers) as pool:
        for path in tracks:
            while len(pending) >= a.workers:  # a track's stems and estimates are ~0.5-1 GB: one in flight per worker
                collect(next(iter(pending)))
            name = os.path.basename(path)
            if name in res:
                continue
            mix, stems = load_track(path)
            est = separate(model, mix.cuda(), batch_size=a.batch_size, overlap=a.overlap, precision=a.precision).cpu()
            res[name] = {"usdr": usdr(stems, est).tolist()}
            pending[name] = pool.submit(bss_eval, stems.numpy(), est.numpy())
            del mix, stems, est
        while pending:
            collect(next(iter(pending)))

    csdr = np.array([r["csdr"] for r in res.values()])
    u = np.array([r["usdr"] for r in res.values()])
    med, mean = np.nanmedian(csdr, 0), u.mean(0)
    print(f"\n{len(res)} test tracks | " + " | ".join(STEMS) + " | avg")
    print("cSDR (median) | " + " | ".join(f"{v:.2f}" for v in med) + f" | {med.mean():.2f}")
    print("uSDR (mean)   | " + " | ".join(f"{v:.2f}" for v in mean) + f" | {mean.mean():.2f}")
    if a.out:
        summary = {"csdr": dict(zip(STEMS, med.tolist())), "csdr_avg": float(med.mean()), "usdr": dict(zip(STEMS, mean.tolist())), "usdr_avg": float(mean.mean())}
        json.dump({"checkpoint": a.weights, "cfg": ck["cfg"], "step": ck.get("step"),
                   "inference": inference, "evaluation": identity,
                   "summary": summary, "tracks": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
