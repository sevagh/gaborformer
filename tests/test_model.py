import os

import pytest
import torch

from gaborformer.model import Gaborformer, count_parameters
from gaborformer.separate import load_model, separate, usdr

LENGTH = 44100
RELEASE = "lin10.77-cq4k-bw43-u16k500@hop1024@timeinv@read10.77"
TINY = dict(frontend=RELEASE, length=LENGTH, dim=32, depth=1, heads=2, dim_head=16, mask_mult=1)


def test_forward_shapes_and_complex_mask_identity():
    m = Gaborformer(**TINY)
    x = torch.randn(2, 2, LENGTH)
    assert m(x).shape == (2, 4, 2, LENGTH)
    # masks of exactly 1 + 0j must give back the mixture for every stem
    C = m.frontend(x)
    assert torch.allclose(m.frontend.inverse(torch.stack([C] * 4, 1)), x[:, None].expand(-1, 4, -1, -1), atol=1e-4)


def test_release_architecture_size():
    m = Gaborformer()
    assert len(m.plan) == 131 and abs(count_parameters(m) / 1e6 - 266.6) < 0.1


def test_analysis_only_bins_get_no_masks():
    m = Gaborformer(**TINY)
    assert sum(m.dim_out) == len(m.synth_index) * 4 < sum(m.dim_in)
    assert m(torch.randn(2, 2, LENGTH)).shape == (2, 4, 2, LENGTH)


class Repeat(torch.nn.Module):
    """Stand-in model: every stem is the input chunk."""

    length, n_stems = LENGTH, 4

    def forward(self, chunk):
        return chunk[:, None].expand(-1, self.n_stems, -1, -1)


def test_separate_overlap_add_is_exact():
    mix = torch.randn(2, 3 * LENGTH + 777)
    for overlap in (0.0, 0.5, 0.75):
        est = separate(Repeat(), mix, batch_size=3, overlap=overlap, precision="fp32")
        assert est.shape == (4, 2, mix.shape[-1])
        assert float(usdr(mix.expand(4, -1, -1), est).min()) > 100, overlap
    with pytest.raises(ValueError):
        separate(Repeat(), mix, overlap=1.0)
    with pytest.raises(ValueError):
        separate(Repeat(), mix, precision="int8")


def test_safetensors_export_roundtrip_and_cfg_keys(tmp_path):
    from scripts.export_weights import export
    cfg = {"frontend": RELEASE, "bands": "budget10-mid-2k65", "length": LENGTH, "dim": 32, "depth": 1, "heads": 2, "dim_head": 16, "mask_depth": 2, "mask_mult": 1}
    torch.manual_seed(0)
    m = Gaborformer.from_config(cfg).eval()
    src = tmp_path / "best.pt"
    torch.save({"cfg": cfg, "model": m.state_dict(), "step": 123, "val": {"avg_mean": 1.5}}, src)
    written, _ = export(str(src), str(tmp_path), "tiny")
    assert [os.path.basename(p) for p, _, _ in written] == ["tiny-fp32.safetensors", "tiny-fp16.safetensors"]
    assert written[1][1] < 0.6 * written[0][1]
    x = torch.randn(1, 2, LENGTH)
    with torch.no_grad():
        ref = m(x)
        for path, _, _ in written:
            loaded, ck = load_model(path, device="cpu")
            assert ck["step"] == 123 and ck["val"]["avg_mean"] == 1.5 and ck["cfg"]["dim"] == 32
            assert next(loaded.parameters()).dtype == torch.float32
            tol = 1e-6 if path.endswith("fp32.safetensors") else 2e-2
            assert torch.allclose(loaded(x), ref, atol=tol, rtol=tol), path
    with pytest.raises(ValueError):
        Gaborformer.from_config({**cfg, "unknown_option": True})


@pytest.mark.parametrize("cache_source", ["default", "xdg", "override"])
def test_compile_cache_is_selected_before_compilation(tmp_path, monkeypatch, cache_source):
    # Patch compilation itself: exercise checkpoint loading and cache selection without building kernels.
    source = Gaborformer(**TINY)
    checkpoint = tmp_path / "tiny.pt"
    torch.save({"cfg": TINY, "model": source.state_dict()}, checkpoint)
    xdg = str(tmp_path / "xdg-cache")
    override = str(tmp_path / "explicit-cache")
    monkeypatch.setenv("XDG_CACHE_HOME", xdg)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", override)
    if cache_source != "override":
        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR")
    if cache_source == "default":
        monkeypatch.delenv("XDG_CACHE_HOME")
    expected = {
        "default": os.path.expanduser("~/.cache/gaborformer/inductor"),
        "xdg": os.path.join(xdg, "gaborformer", "inductor"),
        "override": override,
    }[cache_source]
    observed = []
    monkeypatch.setattr(Gaborformer, "compile_transformers",
                        lambda self: observed.append(os.environ.get("TORCHINDUCTOR_CACHE_DIR")))

    before = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    load_model(checkpoint, device="cpu")
    assert observed == [] and os.environ.get("TORCHINDUCTOR_CACHE_DIR") == before

    loaded, _ = load_model(checkpoint, device="cpu", compile=True)
    assert observed == [expected]
    assert not loaded.training and next(loaded.parameters()).device.type == "cpu"


def test_replan_initializes_from_a_coarser_plan():
    src = Gaborformer(bands="budget10-mid", **TINY)
    dst = Gaborformer(bands="budget10-mid-2k65", **TINY)
    copied, sliced, fresh = dst.load_replan_state_dict(src.state_dict(), src.plan)
    assert fresh == 0 and copied + sliced == len(dst.plan) == 131 and sliced == 32
    assert torch.equal(dst.band_split[0][1].weight, src.band_split[0][1].weight)  # an identical band copies
    # a split band: its mask heads are the source heads restricted to the rows of its own features
    synth = dst.synth_index.tolist()
    i = next(i for i, b in enumerate(dst.plan) if b not in src.plan)
    s, e = dst.plan[i]
    j, (ss, ee) = next((j, b) for j, b in enumerate(src.plan) if b[0] <= s and e <= b[1])
    old = [f for f in synth if ss <= f < ee]
    p, n = sum(f < s for f in old), sum(s <= f < e for f in old)
    z = torch.randn(2, 5, 32)
    for st in range(4):
        a, b = src.mask_estimators[st][j](z), dst.mask_estimators[st][i](z)
        assert torch.allclose(b, a[..., p * 4:(p + n) * 4], atol=1e-6)
    c = (s - ss) * 4
    assert torch.equal(dst.band_split[i][1].weight, src.band_split[j][1].weight[:, c:c + (e - s) * 4])
    y = dst(torch.randn(2, 2, LENGTH))
    assert y.shape == (2, 4, 2, LENGTH) and torch.isfinite(y).all()
