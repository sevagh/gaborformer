import numpy as np
import pytest
import torch

from gaborformer.bands import BAND_PLANS
from gaborformer.nsgt import FILTERBANKS, make_frontend

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RELEASE = "lin10.77-cq4k-bw43-u16k500@hop1024@timeinv@read10.77"
NAMES = [*FILTERBANKS, "lin10.77-cq4k-bw43@hop1024@timeinv", RELEASE]
LENGTH = 132300  # 3 s at 44.1 kHz


def snr_db(x, y):
    return float(10 * torch.log10(x.pow(2).sum() / (x - y).pow(2).sum()))


@pytest.mark.parametrize("name", NAMES)
def test_perfect_reconstruction(name):
    F = make_frontend(name, length=LENGTH).to(DEVICE)
    x = torch.randn(3, 2, LENGTH, device=DEVICE)
    C = F(x)
    assert C.shape == (3, 2, F.n_frames, F.n_features) and C.is_complex()
    y = F.inverse(C)
    assert y.shape == x.shape and snr_db(x, y) > 110


@pytest.mark.parametrize("name", NAMES)
def test_bin_slices_partition_features_in_frequency_order(name):
    F = make_frontend(name, length=LENGTH)
    starts, ends = zip(*F.bin_slices)
    assert starts[0] == 0 and ends[-1] == F.n_features
    assert all(e == s for e, s in zip(ends[:-1], starts[1:]))
    assert np.all(np.diff(F.centres) >= 0)


def test_analysis_only_bins_reconstruct_through_the_rest():
    F = make_frontend(RELEASE, length=LENGTH)
    x = torch.randn(2, LENGTH)
    C = F(x)
    assert C.shape[-1] == F.n_features > len(F.synth_index)
    for coef in (C, C[..., F.synth_index]):
        assert (F.inverse(coef) - x).abs().max() < 1e-4


def test_release_frontend_layout():
    F = make_frontend(RELEASE)
    assert F.length == 352800 and F.n_frames == 345 and F.n_features == 1688 and len(F.synth_index) == 1502
    assert abs(F.redundancy - 1.65) < 0.01


def test_band_plans_partition_features_with_the_documented_token_counts():
    F = make_frontend(RELEASE)
    for name, n in (("budget10", 84), ("budget10-mid", 115), ("budget10-mid-2k65", 131)):
        plan = BAND_PLANS[name](F)
        assert len(plan) == n, name
        assert plan[0][0] == 0 and plan[-1][1] == F.n_features
        assert all(e == s for (_, e), (s, _) in zip(plan[:-1], plan[1:]))
    # the finer plans nest inside the coarser ones (what the sliced warm start relies on)
    coarse, fine = BAND_PLANS["budget10-mid"](F), BAND_PLANS["budget10-mid-2k65"](F)
    assert all(any(ss <= s and e <= ee for ss, ee in coarse) for s, e in fine)


def test_unknown_option_is_rejected():
    with pytest.raises(ValueError):
        make_frontend("lin10.77-cq4k-bw43@unknown", length=LENGTH)
