# Development and verification

[← README](../README.md)

Run these commands from the repository root.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

The tests cover invertibility, feature layouts, mask shapes, checkpoint export/loading, band refinement, and overlap-add reconstruction. For numerical verification of the full released model, a deterministic synthetic clip and reference outputs are included. No dataset audio is redistributed:

```bash
python scripts/reference_output.py weights/gaborformer-fp32.safetensors \
    --check tests/reference
```

`gaborformer/` contains the transform, model, inference, and training implementation. `scripts/` contains evaluation, checkpoint averaging, weight export, and reference verification. `tests/` includes reference outputs and per-track benchmark scores.
