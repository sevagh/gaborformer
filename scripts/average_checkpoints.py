"""Average the EMA weights of several training snapshots into one best.pt-style checkpoint.

Each last.pt carries the EMA parameters (the weights best.pt is written from). Averaging snapshots spread over the
whole learning-rate decay extends the EMA horizon (0.999 decay ~ 1k steps) to that window. The result is one model
with the same architecture and inference cost. Choose snapshots using held-out validation results.

    python scripts/average_checkpoints.py runs/mine/snap-*/last.pt --out runs/mine/avg.pt

Snapshots are read one at a time, so peak RAM is one checkpoint plus the running mean.
"""
import argparse
import gc
import os

import torch

from gaborformer.model import Gaborformer


def ema_params(ck):
    """EMA parameter list in model.parameters() order; the raw weights when a checkpoint has no EMA."""
    if ck.get("ema") is not None:
        return [t.detach().float() for t in ck["ema"]]
    model = Gaborformer.from_config(ck["cfg"])
    model.load_state_dict(ck["model"])
    return [p.detach().float() for p in model.parameters()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("paths", nargs="+", help="last.pt files to average")
    p.add_argument("--out", required=True, help="output checkpoint (best.pt format: cfg, model, step, val)")
    a = p.parse_args(argv)
    acc, steps, last_ck = None, [], None
    for k, path in enumerate(a.paths, 1):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        params = ema_params(ck)
        if acc is None:
            acc = [t.clone() for t in params]
        else:
            assert len(acc) == len(params), "parameter count differs between snapshots"
            torch._foreach_lerp_(acc, params, 1.0 / k)  # running mean
        steps.append(int(ck.get("step", -1)))
        print(f"[{k}/{len(a.paths)}] step {steps[-1]} from {path}", flush=True)
        if k == len(a.paths):
            last_ck = {"cfg": ck["cfg"], "model": ck["model"], "step": ck.get("step")}
        del ck, params
        gc.collect()
    model = Gaborformer.from_config(last_ck["cfg"])
    model.load_state_dict(last_ck["model"])
    with torch.no_grad():
        for prm, avg in zip(model.parameters(), acc):
            prm.copy_(avg.to(prm.dtype))
    state = {"cfg": last_ck["cfg"], "model": model.state_dict(), "step": last_ck["step"], "val": None, "averaged_steps": steps}
    torch.save(state, a.out + ".tmp")
    os.replace(a.out + ".tmp", a.out)
    print(f"wrote {a.out}: EMA average of {len(steps)} snapshots, steps {steps[0]}..{steps[-1]}")


if __name__ == "__main__":
    main()
