"""Export a training checkpoint (best.pt or an averaged checkpoint) to release weight files: safetensors in fp32 and fp16.

    python scripts/export_weights.py runs/mine/best.pt --out-dir weights --name gaborformer-mine

writes <out-dir>/<name>-fp32.safetensors and <name>-fp16.safetensors, each with the model config, training step,
validation score, source checkpoint sha256 and weight dtype in the file's metadata, so gaborformer.separate can load
either directly. fp16 halves the download; the weights are upcast to fp32 at load and the forward pass runs at the
CLI's --precision, so fp16 storage costs nothing at inference.
"""
import argparse
import hashlib
import json
import os

import torch
from safetensors.torch import save_file


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def export(ckpt, out_dir, name, dtypes=("fp32", "fp16")):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    base = {"cfg": json.dumps(ck["cfg"]), "step": json.dumps(ck.get("step")), "val": json.dumps(ck.get("val")),
            "source_sha256": sha256(ckpt), "format": "gaborformer-v1"}
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for d in dtypes:
        torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[d]
        sd = {k: v.detach().to(torch_dtype).contiguous() if v.is_floating_point() else v.detach().contiguous() for k, v in ck["model"].items()}
        path = os.path.join(out_dir, f"{name}-{d}.safetensors")
        save_file(sd, path, metadata={**base, "dtype": d})
        written.append((path, os.path.getsize(path), sha256(path)))
    return written, ck


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoint")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--name", required=True, help="file stem, e.g. gaborformer-mine")
    p.add_argument("--dtypes", default="fp32,fp16")
    a = p.parse_args(argv)
    written, ck = export(a.checkpoint, a.out_dir, a.name, tuple(a.dtypes.split(",")))
    n = sum(v.numel() for v in ck["model"].values())
    print(f"{n / 1e6:.1f}M params, step {ck.get('step')}")
    for path, size, d in written:
        print(f"{path}  {size:,} bytes  sha256 {d}")


if __name__ == "__main__":
    main()
