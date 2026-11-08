"""Gaborformer: a band-split RoPE transformer (BS-RoFormer, Lu et al. 2023) on the NSGT front-end.

Per band, RMSNorm + Linear map the band's complex coefficients (features x channels x re/im) to a token of width
``dim``. ``depth`` pairs of transformers alternate along time (within a band) and across bands (within a frame), with
rotary position embeddings. Per band and per stem, an MLP with a GLU output produces a complex mask over the band's
synthesized features; the masked coefficients are inverted by the NSGT. The structure follows lucidrains/BS-RoFormer
(MIT) without its later additions (hyper-connections, value residuals, attention gates).
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .bands import BAND_PLANS
from .nsgt import make_frontend

STEMS = ("vocals", "drums", "bass", "other")


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class Rotary(nn.Module):
    def __init__(self, dim_head, theta=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / theta ** (torch.arange(0, dim_head, 2).float() / dim_head), persistent=False)

    def forward(self, x):  # x: (..., n, d), rotating interleaved pairs at positions 0, 1, 2, ...
        n = x.shape[-2]
        ang = torch.arange(n, device=x.device, dtype=torch.float32)[:, None] * self.inv_freq
        cos, sin = ang.cos().repeat_interleave(2, -1).to(x.dtype), ang.sin().repeat_interleave(2, -1).to(x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        return x * cos + rot * sin


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, rotary):
        super().__init__()
        self.heads, self.rotary = heads, rotary
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, 3 * heads * dim_head, bias=False)
        self.to_out = nn.Linear(heads * dim_head, dim, bias=False)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.to_qkv(self.norm(x)).view(b, n, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        q, k = self.rotary(q), self.rotary(k)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.to_out(out.transpose(1, 2).reshape(b, n, -1))


class FeedForward(nn.Sequential):
    def __init__(self, dim, mult=4):
        # the two Dropout slots keep the layer indices (and so the state-dict keys) of the training-time module
        super().__init__(RMSNorm(dim), nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(0.0), nn.Linear(dim * mult, dim), nn.Dropout(0.0))


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, rotary):
        super().__init__()
        self.layers = nn.ModuleList(nn.ModuleList([Attention(dim, heads, dim_head, rotary), FeedForward(dim)]) for _ in range(depth))

    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return x


def mlp(dim_in, dim_out, dim_hidden, depth):
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


# Architecture fields stored in release and training checkpoints.
MODEL_KEYS = ("frontend", "bands", "length", "dim", "depth", "heads", "dim_head", "mask_depth", "mask_mult")


class Gaborformer(nn.Module):
    """audio (batch, channels, length) -> stems (batch, n_stems, channels, length)."""

    def __init__(self, frontend="lin10.77-cq4k-bw43-u16k500@hop1024@timeinv@read10.77", bands="budget10-mid-2k65", length=352800, channels=2,
                 n_stems=4, dim=512, depth=16, heads=8, dim_head=64, mask_depth=2, mask_mult=1, checkpointing=False):
        super().__init__()
        self.checkpointing = checkpointing
        self.frontend = make_frontend(frontend, length=length)
        self.length, self.channels, self.n_stems = length, channels, n_stems
        self.plan = BAND_PLANS[bands](self.frontend)
        self.dim_in = [(e - s) * channels * 2 for s, e in self.plan]  # features x channels x (re, im)
        synth = self.frontend.synth_index
        if synth is None:
            self.dim_out = self.dim_in
        else:  # masks only for the synthesized features of each band
            self.register_buffer("synth_index", synth.clone(), persistent=False)
            self.dim_out = [int(((synth >= s) & (synth < e)).sum()) * channels * 2 for s, e in self.plan]
            assert min(self.dim_out) > 0, "every band needs a synthesized bin"
        self.band_split = nn.ModuleList(nn.Sequential(RMSNorm(d), nn.Linear(d, dim)) for d in self.dim_in)
        rot_t, rot_b = Rotary(dim_head), Rotary(dim_head)
        self.layers = nn.ModuleList(
            nn.ModuleList([Transformer(dim, 1, heads, dim_head, rot_t), Transformer(dim, 1, heads, dim_head, rot_b)]) for _ in range(depth)
        )
        self.final_norm = RMSNorm(dim)
        # per stem, per band: MLP to 2x features, GLU -> the band's complex mask (re, im per feature and channel)
        self.mask_estimators = nn.ModuleList(
            nn.ModuleList(nn.Sequential(mlp(dim, d * 2, dim * mask_mult, mask_depth), nn.GLU(dim=-1)) for d in self.dim_out) for _ in range(n_stems)
        )

    @classmethod
    def from_config(cls, cfg, **kwargs):
        """Build from a checkpoint's cfg dict (as written by train.py and carried in the safetensors metadata)."""
        unknown = set(cfg) - set(MODEL_KEYS)
        if unknown:
            raise ValueError(f"unsupported model configuration fields: {', '.join(sorted(unknown))}")
        args = {k: cfg[k] for k in MODEL_KEYS if k in cfg}
        if not isinstance(args.get("mask_mult", 1), int):
            raise ValueError(f"per-stem mask widths ({cfg['mask_mult']!r}) are not supported here")
        return cls(**args, **kwargs)

    def _to_bands(self, C):
        # (b, ch, t, f) complex -> tuple of (b, t, (f, ch, re/im)) per band
        x = torch.view_as_real(C).permute(0, 2, 3, 1, 4)  # b t f ch 2
        return x.reshape(*x.shape[:2], -1).split(self.dim_in, dim=-1)

    def forward(self, audio):
        b, ch, n = audio.shape
        C = self.frontend(audio.float())  # complex (b, ch, t, f), fp32 outside autocast
        t = C.shape[2]
        x = torch.stack([split(band) for split, band in zip(self.band_split, self._to_bands(C))], dim=2)  # b t k d
        k = x.shape[2]
        run = (lambda f, y: checkpoint(f, y, use_reentrant=False)) if self.checkpointing and self.training else (lambda f, y: f(y))
        for time_tf, band_tf in self.layers:
            x = run(time_tf, x.transpose(1, 2).reshape(b * k, t, -1)).view(b, k, t, -1).transpose(1, 2)
            x = run(band_tf, x.reshape(b * t, k, -1)).view(b, t, k, -1)
        x = self.final_norm(x)
        if hasattr(self, "synth_index"):
            C = C[..., self.synth_index]
        tokens = x.unbind(dim=2)
        out = []
        for estimators in self.mask_estimators:
            m = torch.cat([est(z) for z, est in zip(tokens, estimators)], dim=-1)  # b t (f ch 2)
            m = m.float().view(b, t, -1, ch, 2).permute(0, 3, 1, 2, 4)  # b ch t f 2
            out.append(C * torch.view_as_complex(m.contiguous()))
        return self.frontend.inverse(torch.stack(out, dim=1))  # b stems ch t f -> b stems ch n

    def load_replan_state_dict(self, sd, src_plan):
        """Warm start from weights trained with a different band plan on the same front-end. A band whose feature
        range equals a source band's copies that band; a band that lies inside one source band gets the slice of that
        band's band-split input (RMSNorm gamma and Linear columns of its own features) and, per stem, the source mask
        head with only the output rows (GLU value and gate halves) of its own synthesized features; any other band
        keeps its fresh initialization. The trunk loads unchanged. Returns (copied, sliced, fresh) band counts."""
        c2 = self.channels * 2
        synth = self.synth_index.tolist() if hasattr(self, "synth_index") else list(range(self.frontend.n_features))
        new = {k: v for k, v in sd.items() if not k.startswith(("band_split.", "mask_estimators."))}
        counts = [0, 0, 0]
        for i, (s, e) in enumerate(self.plan):
            j = next((j for j, (ss, ee) in enumerate(src_plan) if ss <= s and e <= ee), None)
            if j is None:
                counts[2] += 1
                continue
            ss, ee = src_plan[j]
            counts[0 if (ss, ee) == (s, e) else 1] += 1
            a, b = (s - ss) * c2, (e - ss) * c2  # this band's input features inside the source band's
            new[f"band_split.{i}.0.gamma"] = sd[f"band_split.{j}.0.gamma"][a:b]
            new[f"band_split.{i}.1.weight"] = sd[f"band_split.{j}.1.weight"][:, a:b]
            new[f"band_split.{i}.1.bias"] = sd[f"band_split.{j}.1.bias"]
            old_synth = [f for f in synth if ss <= f < ee]
            p, n = sum(f < s for f in old_synth), sum(s <= f < e for f in old_synth)
            d_old = len(old_synth) * c2
            for st in range(self.n_stems):
                src, dst = f"mask_estimators.{st}.{j}.0.", f"mask_estimators.{st}.{i}.0."
                last = f"{src}{len(self.mask_estimators[st][i][0]) - 1}."
                for key in [key for key in sd if key.startswith(src)]:
                    v = sd[key]
                    if key.startswith(last):  # output layer: value rows then gate rows, this band's features only
                        v = torch.cat([v[p * c2:(p + n) * c2], v[d_old + p * c2:d_old + (p + n) * c2]], dim=0)
                    new[key.replace(src, dst, 1)] = v
        own = self.state_dict()
        self.load_state_dict({**own, **{k: v for k, v in new.items() if k in own}})
        return tuple(counts)

    def compile_transformers(self):
        """torch.compile each transformer in place (state-dict keys unchanged); ~1.4x faster inference and training."""
        for pair in self.layers:
            for tf in pair:
                tf.compile()
        return self


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
