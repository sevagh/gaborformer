"""Train or fine-tune a Gaborformer on MUSDB18-HQ (or any folder of tracks in its layout), validating on the standard
14-track validation split.

Resumable: a run directory holds last.pt (every --ckpt-every steps), best.pt (best validation uSDR of the EMA
weights) and log.jsonl. Rerunning with the same --run-dir continues the run. Single GPU, or data parallel under
torchrun (--batch-size is the global batch, split across ranks).

The defaults are the released recipe's architecture (dim 512, depth 16, 131 band tokens, 266.6M parameters). The
released model was trained in stages: 400k steps from scratch with the 84-token plan at lr 3e-4 (4k warmup, cosine
to 3e-5), then warm restarts into finer plans at lr 1e-4 (1k warmup, cosine to 1e-5): 40k + 100k steps with
budget10-mid, 60k with budget10-mid-2k65. The final weights average 14 EMA snapshots spread across the last stage's
decay (scripts/average_checkpoints.py). Training uses CUDA with bf16 autocast and activation checkpointing.

    python -m gaborformer.train --musdb /data/musdb18hq --run-dir runs/mine --steps 400000 --bands budget10 --lr 3e-4 --warmup 4000
    python -m gaborformer.train --musdb /data/musdb18hq --run-dir runs/finetune --init-from weights/gaborformer-fp32.safetensors \\
        --steps 60000 --lr 1e-4
"""
import argparse
import json
import math
import os
import signal
import sys
import time
import traceback

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .bands import BAND_PLANS
from .data import RandomMix, list_tracks, load_track, musdb_root
from .losses import separation_loss
from .model import MODEL_KEYS, STEMS, Gaborformer, count_parameters
from .separate import read_checkpoint, separate, usdr


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--musdb", help="MUSDB18-HQ root (train/ and test/ folders of track folders); default $MUSDB_PATH")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--frontend", default="lin10.77-cq4k-bw43-u16k500@hop1024@timeinv@read10.77")
    p.add_argument("--bands", default="budget10-mid-2k65", choices=tuple(BAND_PLANS), help="band plan (gaborformer/bands.py)")
    p.add_argument("--length", type=int, default=352800, help="training chunk in samples (8 s); validation separates with the same chunk")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dim-head", type=int, default=64)
    p.add_argument("--mask-depth", type=int, default=2)
    p.add_argument("--mask-mult", type=int, default=1, help="mask-head hidden width = dim x this")
    p.add_argument("--init-from", help="warm start from these weights (.safetensors or .pt, EMA weights); a different band plan on the same "
                                       "front-end is sliced into place (Gaborformer.load_replan_state_dict)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1, help="the cosine floor as a fraction of --lr")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--steps", type=int, default=60000, help="total optimizer steps (cosine schedule length)")
    p.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay on matrices (not norms or biases)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--max-hours", type=float, default=None, help="stop (resumably) after this much wall time")
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-compile", action="store_true", help="skip torch.compile of the transformers (~1.6x slower steps)")
    return p.parse_args(argv)


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    t = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * t)))


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.params = [p.detach().clone() for p in model.parameters()]

    @torch.no_grad()
    def update(self, model):
        torch._foreach_lerp_(self.params, [p.detach() for p in model.parameters()], 1 - self.decay)

    @torch.no_grad()
    def swap(self, model):
        """Exchange model weights with the EMA weights (call twice to restore)."""
        for p, e in zip(model.parameters(), self.params):
            tmp = p.detach().clone()
            p.copy_(e)
            e.copy_(tmp)


@torch.no_grad()
def validate(model, tracks):
    model.eval()
    scores = []
    for path in tracks:
        mix, stems = load_track(path)
        est = separate(model, mix.cuda())
        scores.append(usdr(stems.cuda(), est).cpu())
    model.train()
    s = torch.stack(scores)  # tracks x stems
    med, mean = s.median(0).values, s.mean(0)
    out = {f"{n}_median": float(med[i]) for i, n in enumerate(STEMS)}
    out.update({f"{n}_mean": float(mean[i]) for i, n in enumerate(STEMS)})
    out["avg_median"], out["avg_mean"] = float(med.mean()), float(mean.mean())
    out["tracks"] = {os.path.basename(p): {n: float(s[t, i]) for i, n in enumerate(STEMS)} for t, p in enumerate(tracks)}
    return out


def _ignore_sigterm(_worker_id):
    # torchrun signals the whole process group; only the training ranks should react (by checkpointing)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def save(path, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def warm_start(model, cfg, path):
    """--init-from: load EMA weights from a checkpoint with the same architecture, slicing a different band plan into
    place. Returns a description for the log."""
    ck = read_checkpoint(path)
    src = {k: ck["cfg"].get(k) for k in MODEL_KEYS}
    same = {k: v for k, v in src.items() if k != "bands"}
    assert same == {k: cfg[k] for k in same}, f"{path} was trained with {ck['cfg']}; only the band plan may differ"
    if src["bands"] == cfg["bands"]:
        model.load_state_dict(ck["model"])
        return f"initialized from {path} (step {ck.get('step')})"
    copied, sliced, fresh = model.load_replan_state_dict(ck["model"], BAND_PLANS[src["bands"]](model.frontend))
    return f"initialized from {path} (step {ck.get('step')}); band plan {src['bands']} -> {cfg['bands']}: {copied} bands copied, {sliced} sliced, {fresh} fresh"


def main(argv=None):
    a = parse_args(argv)
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    main_rank = rank == 0
    assert a.batch_size % world == 0, f"--batch-size {a.batch_size} must divide across {world} GPUs"
    musdb = musdb_root(a.musdb)
    os.makedirs(a.run_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    cfg = {k: getattr(a, k) for k in MODEL_KEYS}
    model = Gaborformer(**cfg, checkpointing=True).cuda().train()
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.999))
    ema = EMA(model, a.ema) if main_rank else None
    step, best = 0, -float("inf")

    last = os.path.join(a.run_dir, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cuda", weights_only=False)
        assert ck["cfg"] == cfg, f"run dir was created with {ck['cfg']}"
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if main_rank:
            ema.params = [e.cuda() for e in ck["ema"]]
        step, best = ck["step"], ck["best"]
        del ck
        if main_rank:
            print(f"resumed {a.run_dir} at step {step}", flush=True)
    else:
        if a.init_from:
            how = warm_start(model, cfg, a.init_from)
            if main_rank:
                ema.params = [p.detach().clone() for p in model.parameters()]
                print(how, flush=True)
        if main_rank:
            json.dump({"cfg": cfg, "args": vars(a)}, open(os.path.join(a.run_dir, "config.json"), "w"), indent=1)
    if main_rank:
        print(f"{a.frontend} / {a.bands}: {count_parameters(model) / 1e6:.1f}M params, {len(model.plan)} bands, "
              f"{model.frontend.n_frames} frames x {model.frontend.n_features} features per {model.length / 44100:.0f} s chunk, "
              f"{world} GPU(s) x batch {a.batch_size // world}", flush=True)
    if not a.no_compile:
        model.compile_transformers()
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()]) if ddp else model

    data = RandomMix(musdb, length=model.length, samples_per_epoch=10**9)
    torch.manual_seed(a.seed + step + 1_000_003 * rank)  # fresh, reproducible, per-rank data stream after a resume
    loader = iter(DataLoader(data, batch_size=a.batch_size // world, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4, worker_init_fn=_ignore_sigterm))
    val_tracks = list_tracks(musdb, "valid")
    log = open(os.path.join(a.run_dir, "log.jsonl"), "a") if main_rank else None
    t_start, t_log, acc = time.time(), time.time(), {"loss": 0.0, "l1": 0.0, "mrstft": 0.0, "gnorm": 0.0, "n": 0, "skipped": 0}
    stop = []  # SIGTERM: checkpoint after the current step and exit resumably
    signal.signal(signal.SIGTERM, lambda *_: stop.append(True))
    if ddp:
        dist.barrier()

    while step < a.steps:
        mix, stems = (t.cuda(non_blocking=True) for t in next(loader))
        lr = lr_at(step, a)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.bfloat16):
            est = net(mix)
        loss, (l1, mr) = separation_loss(est, stems)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        if bool(torch.isfinite(gnorm)):  # a non-finite step would poison the weights for good: skip it
            opt.step()
            if main_rank:
                ema.update(model)
            for k, v in (("loss", loss), ("l1", l1), ("mrstft", mr), ("gnorm", gnorm)):
                acc[k] += float(v.detach())
            acc["n"] += 1
        else:
            acc["skipped"] += 1
        opt.zero_grad(set_to_none=True)
        step += 1

        if main_rank and step % a.log_every == 0:
            n = max(1, acc["n"])
            rec = {"step": step, "lr": lr, "sec_per_step": (time.time() - t_log) / (acc["n"] + acc["skipped"]), **{k: acc[k] / n for k in ("loss", "l1", "mrstft", "gnorm")}}
            if acc["skipped"]:
                rec["skipped"] = acc["skipped"]
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(" ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()), flush=True)
        if step % a.log_every == 0:
            t_log, acc = time.time(), {k: 0.0 for k in acc} | {"n": 0, "skipped": 0}

        out_of_time = bool(stop) or (a.max_hours is not None and time.time() - t_start > a.max_hours * 3600)
        if ddp:  # every rank must leave the loop at the same step
            flag = torch.tensor(float(out_of_time), device="cuda")
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            out_of_time = bool(flag.item())
        if main_rank and (step % a.val_every == 0 or step == a.steps):
            ema.swap(model)
            t0 = time.time()
            v = validate(model, val_tracks)
            if v["avg_mean"] > best:
                best = v["avg_mean"]
                save(os.path.join(a.run_dir, "best.pt"), cfg=cfg, model=model.state_dict(), step=step, val=v)
            ema.swap(model)
            log.write(json.dumps({"step": step, "val": v, "val_sec": time.time() - t0, "best_avg_mean": best}) + "\n")
            log.flush()
            print(f"VAL step {step}: uSDR mean {v['avg_mean']:.3f} median {v['avg_median']:.3f} | "
                  + " ".join(f"{n} {v[n + '_mean']:.2f}" for n in STEMS) + f" | best {best:.3f}", flush=True)
        if main_rank and (step % a.ckpt_every == 0 or step == a.steps or out_of_time):
            save(last, cfg=cfg, model=model.state_dict(), opt=opt.state_dict(), ema=ema.params, step=step, best=best)
        if ddp and (step % a.val_every == 0 or step % a.ckpt_every == 0 or step == a.steps or out_of_time):
            dist.barrier()
        if out_of_time:
            if main_rank:
                print(f"stopping at step {step} ({'stop requested' if stop else '--max-hours reached'}); resume with the same command", flush=True)
            break
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # a plain raise can leave the process alive (DataLoader workers), idling a billed GPU: exit hard
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
