"""The invertible nonstationary Gabor transform (NSGT) front-end.

A real signal (..., length) maps to a complex matrix (..., n_frames, n_features) and back. Band k is a Hann window of
bandwidth B_k Hz centred on f_k; its coefficients are sampled at r_k * n_frames points over the signal, so frame t
holds r_k consecutive coefficients of band k. Wide (treble) bands fold their extra time resolution into features and
every band shares one time grid. Each band is left at its own frequency rather than demodulated (coefficient m is
multiplied by exp(2 pi i f_k t_m)): the STFT's time-invariant phase convention, so a sinusoid has the same phase
relation across neighbouring bins in every frame, as in an STFT. ``bin_slices`` gives each bin's [start, end) range
on the feature axis, so a band plan can group contiguous frequency regions.

The transform is circular over its fixed length: build one per chunk length and crossfade chunks (separate.py).

The released model's front-end, ``lin10.77-cq4k-bw43-u16k500@hop1024@timeinv@read10.77``, is a union of two frames:
STFT-4096-like bins (43 Hz Hann windows every 10.77 Hz) below 4 kHz, constant-Q bins above (the same 43 Hz window at
4 kHz, widening with frequency), plus a second layer of fine 10.8 Hz windows every 2.7 Hz below 500 Hz. The fine
bins are analysis-only (``@read10.77``): the model reads them, but masks are produced for, and the inverse
synthesizes through, the other bins only. Frames are 1024 samples apart (``@hop1024``).
"""
import numpy as np
import torch
from torch import nn

SAMPLE_RATE = 44100.0


def lin_cq_centres(df, fc):
    """Linear spacing df up to fc, then constant-Q (spacing f df / fc, continuous at fc) up to Nyquist."""
    f = list(np.arange(df, fc + 1e-9, df))
    while f[-1] * (1 + df / fc) < SAMPLE_RATE / 2 - f[-1] * df / fc / 2:
        f.append(f[-1] * (1 + df / fc))
    return np.array(f)


def band_layout(centres, floor=0.0, bandwidths=None):
    """Positive-frequency bands: DC, the centres, Nyquist. B_k = max(f_{k+1} - f_{k-1}, floor), or max(bandwidths_k,
    floor) when bandwidths are given; centres may then repeat (a union of two frames)."""
    nyq = SAMPLE_RATE / 2
    f = np.asarray(centres, dtype=np.float64)
    assert np.all(np.diff(f) >= 0 if bandwidths is not None else np.diff(f) > 0) and f[0] > 0 and f[-1] < nyq
    ext = np.concatenate([[0.0], f, [nyq]])
    bw = (ext[2:] - ext[:-2]) if bandwidths is None else np.asarray(bandwidths, dtype=np.float64)
    return ext, np.concatenate([[2 * f[0]], np.maximum(bw, floor), [2 * (nyq - f[-1])]])


def union_centres(base, extra_spacing, extra_bw, lo, hi):
    """A second set of bands (extra_bw wide, every extra_spacing Hz) over [lo, hi) added to a base filterbank: two
    time-frequency resolutions of the same region in one frame. -> (centres, bandwidths), sorted, base first among
    equal centres."""
    fc, bw = band_layout(base[0](), base[1])
    fc, bw = fc[1:-1], bw[1:-1]
    extra = np.arange(np.ceil(lo / extra_spacing) * extra_spacing, hi, extra_spacing)
    c = np.concatenate([fc, extra])
    b = np.concatenate([bw, np.full(len(extra), extra_bw)])
    order = np.argsort(c, kind="stable")
    return c[order], b[order]


# name -> (centres, bandwidth floor in Hz)
FILTERBANKS = {
    # STFT-4096 below 4 kHz (43 Hz Hann bands every 10.77 Hz); above, constant-Q bands that widen with frequency
    "lin10.77-cq4k-bw43": (lambda: lin_cq_centres(SAMPLE_RATE / 4096, 4000.0), 43.07),
    # the same plus STFT-16384-like bands (10.8 Hz Hann every 2.7 Hz) below 500 Hz, in one frame
    "lin10.77-cq4k-bw43-u16k500": (lambda: union_centres(FILTERBANKS["lin10.77-cq4k-bw43"], SAMPLE_RATE / 16384, 10.77, SAMPLE_RATE / 16384, 500.0), 0.0),
}


class _Permute(torch.autograd.Function):
    """x[..., perm] whose backward is a gather by the inverse permutation (plain indexing backpropagates through a
    sort-based scatter, several ms per training step)."""

    @staticmethod
    def forward(ctx, x, perm, inv):
        ctx.save_for_backward(inv)
        return x.index_select(-1, perm)

    @staticmethod
    def backward(ctx, g):
        (inv,) = ctx.saved_tensors
        return g.index_select(-1, inv), None, None


class NSGT(nn.Module):
    def __init__(self, centres, floor=0.0, length=352800, n_frames=None, phase="timeinv", read_bw=None):
        super().__init__()
        assert phase in ("baseband", "timeinv")
        self.phase = phase
        n = length
        self.length = n
        self.n_frames = n_frames = n // 512 + 1 if n_frames is None else n_frames
        centres, bandwidths = centres if isinstance(centres, tuple) else (centres, None)
        fc, bw = band_layout(centres, floor, bandwidths)
        self.centres, self.bandwidths = fc, bw
        half = n // 2
        # read_bw: bins at most read_bw Hz wide are analysis-only. They appear in forward()'s features, but inverse()
        # synthesizes through the other bins alone, with the painless dual of that smaller frame.
        synth = np.ones(len(fc), dtype=bool) if read_bw is None else bw > read_bw + 1e-6

        # window support L_k in FFT bins; coefficient count M_k = r_k * n_frames >= L_k
        win = np.maximum(np.round(bw * n / SAMPLE_RATE).astype(np.int64), 4)
        ratio = np.ceil(win / n_frames).astype(np.int64)
        centre_bins = np.round(fc * n / SAMPLE_RATE).astype(np.int64)
        self.ratios = ratio

        groups = {}  # bands sharing a coefficient count share one gather (window padded with zeros)
        for k, r in enumerate(ratio):
            groups.setdefault(int(r), []).append(k)

        den = np.zeros(half + 1)
        specs = []
        for r, bands in sorted(groups.items()):
            span = int(win[bands].max())
            offs = np.arange(span) - span // 2
            idx = (centre_bins[bands, None] + offs[None, :]) % n
            g = np.zeros((len(bands), span))
            for i, k in enumerate(bands):
                L = int(win[k])
                m = offs + L // 2
                inside = (m >= 0) & (m < L)
                g[i, inside] = 0.5 - 0.5 * np.cos(2 * np.pi * m[inside] / L)
            pos = idx <= half  # negative frequencies are implied by realness
            np.add.at(den, idx[pos], (g * synth[bands, None])[pos] ** 2)
            specs.append((r, bands, offs, idx, g, pos))
        assert den.min() > 1e-6, f"frame not covered on [0, Nyquist] (min diagonal {den.min():.3g})"

        # feature order on the rasterized axis: bins ascending in frequency, r_k features each
        starts = np.concatenate([[0], np.cumsum(ratio)])
        self.bin_slices = [(int(starts[k]), int(starts[k + 1])) for k in range(len(ratio))]
        self.n_features = int(starts[-1])
        perm = []  # position in concatenated group outputs -> feature index
        self._groups = []
        for gi, (r, bands, offs, idx, g, pos) in enumerate(specs):
            M = r * n_frames
            neg = int((offs < 0).sum())
            gd = np.where(pos & synth[bands, None], g / np.where(pos, den[np.minimum(idx, half)], 1.0), 0.0)
            self.register_buffer(f"idx{gi}", torch.as_tensor(idx.reshape(-1)), persistent=False)
            self.register_buffer(f"win{gi}", torch.as_tensor(g, dtype=torch.float32), persistent=False)
            self.register_buffer(f"dual{gi}", torch.as_tensor(gd, dtype=torch.float32), persistent=False)
            # synthesis touches only [0, Nyquist] (irfft): keep the window entries there and their target bins
            keep = pos.reshape(-1)
            self.register_buffer(f"keep{gi}", None if keep.all() else torch.as_tensor(np.flatnonzero(keep)), persistent=False)
            self.register_buffer(f"idxh{gi}", torch.as_tensor(idx.reshape(-1)[keep]), persistent=False)
            if phase == "timeinv":  # exp(2 pi i f_k t_m) = exp(2 pi i centre_bin m / M), exact in integers
                turns = (centre_bins[bands, None] % M) * np.arange(M)[None, :] % M
                self.register_buffer(f"ph{gi}", torch.polar(torch.ones(()), torch.as_tensor(2 * np.pi * turns / M, dtype=torch.float32)), persistent=False)
            self._groups.append((len(bands), g.shape[1], neg, M, r))
            for k in bands:
                perm.extend(range(starts[k], starts[k + 1]))
        inv = np.empty(self.n_features, dtype=np.int64)
        inv[np.asarray(perm)] = np.arange(self.n_features)
        self.register_buffer("feat_order", torch.as_tensor(inv), persistent=False)
        self.register_buffer("group_order", torch.as_tensor(np.asarray(perm)), persistent=False)
        # feature indices of the synthesized bins, or None when every bin is synthesized
        self.synth_index = None if synth.all() else torch.as_tensor(np.concatenate([np.arange(*self.bin_slices[k]) for k in np.flatnonzero(synth)]))

    @classmethod
    def from_name(cls, name, **kwargs):
        centres, floor = FILTERBANKS[name]
        return cls(centres(), floor=floor, **kwargs)

    @property
    def redundancy(self):
        """Complex coefficients per input sample (an STFT with n_fft / hop = 4 gives 2.0)."""
        return self.n_features * self.n_frames / self.length

    def forward(self, x):
        assert x.shape[-1] == self.length, f"expected length {self.length}, got {x.shape[-1]}"
        X = torch.fft.fft(x)
        lead = X.shape[:-1]
        out = []
        for gi, (nb, span, neg, M, r) in enumerate(self._groups):
            Y = X[..., getattr(self, f"idx{gi}")].reshape(*lead, nb, span) * getattr(self, f"win{gi}")
            # centre bin at index 0 so a band's coefficients are its baseband (demodulated) signal
            Y = torch.cat([Y[..., neg:], Y.new_zeros(*lead, nb, M - span), Y[..., :neg]], dim=-1)
            c = torch.fft.ifft(Y)  # (..., nb, r * n_frames)
            if self.phase == "timeinv":
                c = c * getattr(self, f"ph{gi}")
            out.append(c.reshape(*lead, nb, self.n_frames, r).transpose(-3, -2).reshape(*lead, self.n_frames, nb * r))
        return _Permute.apply(torch.cat(out, dim=-1), self.feat_order, self.group_order)

    def inverse(self, C, length=None):
        """C: all features, or (with analysis-only bins) just the synth_index ones."""
        assert length in (None, self.length)
        lead = C.shape[:-2]
        if self.synth_index is not None and C.shape[-1] == len(self.synth_index):
            C = C.new_zeros(*C.shape[:-1], self.n_features).index_copy(-1, self.synth_index.to(C.device), C)
        C = _Permute.apply(C, self.group_order, self.feat_order)
        Xh = C.new_zeros(*lead, self.length // 2 + 1)
        start = 0
        for gi, (nb, span, neg, M, r) in enumerate(self._groups):
            c = C[..., start:start + nb * r]
            start += nb * r
            c = c.reshape(*lead, self.n_frames, nb, r).transpose(-3, -2).reshape(*lead, nb, M)
            if self.phase == "timeinv":
                c = c * getattr(self, f"ph{gi}").conj()
            Y = torch.fft.fft(c)
            Y = torch.cat([Y[..., M - neg:], Y[..., :span - neg]], dim=-1) * getattr(self, f"dual{gi}")
            Y, keep = Y.reshape(*lead, nb * span), getattr(self, f"keep{gi}")
            Xh.index_add_(-1, getattr(self, f"idxh{gi}"), Y if keep is None else Y.index_select(-1, keep))
        return torch.fft.irfft(Xh, n=self.length)


def make_frontend(name, length=352800):
    """A FILTERBANKS key with optional '@hop<samples>' (frame spacing; default 512), '@timeinv' (the STFT's phase
    convention; the default is baseband, i.e. demodulated bands) and '@read<bw>' (bins at most <bw> Hz wide are
    analysis-only) suffixes, e.g. the release front-end 'lin10.77-cq4k-bw43-u16k500@hop1024@timeinv@read10.77'."""
    name, *opts = name.split("@")
    kwargs = {"phase": "baseband"}
    for o in opts:
        if o.startswith("hop"):
            kwargs["n_frames"] = length // int(o[3:]) + 1
        elif o in ("timeinv", "local"):  # "local" is the option's old name, kept for old checkpoints
            kwargs["phase"] = "timeinv"
        elif o.startswith("read"):
            kwargs["read_bw"] = float(o[4:])
        else:
            raise ValueError(f"unknown front-end option {o!r}")
    return NSGT.from_name(name, length=length, **kwargs)
