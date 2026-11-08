"""Band plans: which contiguous feature ranges of the front-end form each band token.

A plan is a list of band edges in Hz. Every front-end bin goes to the band its centre falls in; a band holds all of
its bins' features (r_k per bin). The plans below were placed by the trained models' error budget (where the error
energy is and where the learned masks trail the ideal-mask oracle), starting from BS-RoFormer's 62-band split of a
2048-point STFT:

* ``budget10`` (84 tokens): one 10.8 Hz bin per token from 27 to 490 Hz, BS-RoFormer's own edges from 506 Hz to
  4.1 kHz, 7 treble bands. Used for the first training stage.
* ``budget10-mid`` (115): the same bass tokens; 43 Hz tokens to 2.07 kHz, 129 Hz to 4.1 kHz, 517 Hz to 8.3 kHz, 1 kHz
  to 16.5 kHz, 2 treble bands.
* ``budget10-mid-2k65`` (131): ``budget10-mid`` with its 2.07-4.1 kHz tokens halved to 65 Hz. The released model.

A model trained with one plan can be warm-started into a finer one that nests inside it (``Gaborformer.load_replan_state_dict``).
"""
import numpy as np

from .nsgt import SAMPLE_RATE

# BS-RoFormer (Lu et al. 2023) bins per band for n_fft=2048
BS_ROFORMER_BINS = (2,) * 24 + (4,) * 12 + (12,) * 8 + (24,) * 8 + (48,) * 8 + (128, 129)


def bs_roformer_edges_hz():
    """Upper edge of each BS-RoFormer band in Hz (bin boundaries of the 2048-point STFT, midway between bins)."""
    return (np.cumsum(BS_ROFORMER_BINS) - 0.5) * SAMPLE_RATE / 2048


def band_plan(frontend, edges_hz):
    """List of (start, end) feature ranges, one per band, in ascending frequency. A band that would hold no bins is
    merged into its neighbour."""
    edges = np.asarray(edges_hz)
    band_of_bin = np.minimum(np.searchsorted(edges, frontend.centres), len(edges) - 1)
    plan, start_bin = [], 0
    for k in range(1, len(band_of_bin) + 1):
        if k == len(band_of_bin) or band_of_bin[k] != band_of_bin[k - 1]:
            plan.append((frontend.bin_slices[start_bin][0], frontend.bin_slices[k - 1][1]))
            start_bin = k
    return plan


def _edges(*parts):
    e = np.unique(np.concatenate([np.asarray(x, dtype=float) for x in parts]))
    assert np.all(np.diff(e) > 1.0), "band edges must be strictly increasing (at least 1 Hz apart)"
    return e


def budget10_edges_hz():
    std = bs_roformer_edges_hz()
    lo = (np.arange(3, 47) - 0.5) * SAMPLE_RATE / 4096  # 27 ... 490 Hz, one 10.8 Hz bin per token
    return np.concatenate([lo, std[11:44], std[[45, 47, 49, 51, 55, 59, 61]]])


def budget10_mid_edges_hz(mid_bins=6):
    """budget10's bass tokens; 43 Hz tokens to 2.07 kHz; `mid_bins` STFT-2048 bins (21.5 Hz each) per token to
    4.1 kHz; 517 Hz tokens to 8.3 kHz, 1 kHz to 16.5 kHz, 2 treble bands."""
    df = SAMPLE_RATE / 2048
    lo = (np.arange(3, 47) - 0.5) * SAMPLE_RATE / 4096
    mid = (np.arange(24, 97, 2) - 0.5) * df  # 506 ... 2078 Hz every 43 Hz
    upper = (np.arange(96 + mid_bins, 193, mid_bins) - 0.5) * df  # 2.1 ... 4.1 kHz
    std = bs_roformer_edges_hz()
    return _edges(lo, mid, upper, std[44:52], std[52:60], std[[60, 61]])


BAND_PLANS = {
    "budget10": lambda frontend: band_plan(frontend, budget10_edges_hz()),
    "budget10-mid": lambda frontend: band_plan(frontend, budget10_mid_edges_hz(6)),
    "budget10-mid-2k65": lambda frontend: band_plan(frontend, budget10_mid_edges_hz(3)),
}
