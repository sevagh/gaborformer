# Architecture and training

[← README](../README.md)

## Architecture

Gaborformer revisits the idea behind my 2021 master's work, [xumx-sliCQ][xumx]: use the NSGT as the time-frequency representation for music separation. The new model brings that representation to a modern transformer that exchanges information across both time and frequency.

```mermaid
flowchart LR
    A["Stereo audio · 8 s"] --> B["Multi-resolution NSGT"]
    B --> C["131 band tokens"]
    C --> D["Time and band transformers"]
    D --> E["Four complex masks"]
    B --> F["Mask and inverse NSGT"]
    E --> F
    F --> G["Vocals · drums · bass · other"]
```

The front-end uses **different resolutions at different frequencies**: linear Hann bands below 4 kHz, constant-Q bands above 4 kHz, and an additional fine-resolution layer below 500 Hz. The fine bass bins are analysis-only: they inform the network while reconstruction uses the main filterbank. The transform uses the STFT-compatible time-invariant phase convention and has an exact inverse up to numerical precision.

An 8-second chunk becomes 345 frames of 1,688 complex features. These are grouped into **131 frequency-band tokens**, with dense bass coverage and finer tokens through the midrange. Sixteen pairs of time and band transformers, at width 512 with eight attention heads, predict complex masks for the four stems. The masks act on 1,502 synthesis features, and inverse NSGT reconstructs each waveform. Adjacent chunks are joined with triangular crossfades at 50% overlap.

The architecture builds on [BS-RoFormer][bsrf] and [Phil Wang's implementation][lucidrains], with the [nonstationary Gabor frame][nsgt-paper] as its spectral representation. It continues the spectral-masking approach of [xumx-sliCQ (Hanssian, 2021)][xumx-paper], which used the [sliCQ framework of Holighaus et al. (2013)][slicq-paper]. The headline result measures the complete system.

### Model size

Gaborformer has **266.6M parameters**: approximately **100.8M in the transformer trunk**, **162.3M in the band-wise mask heads**, and **3.5M in the band-split projections**. The mask heads are MLPs applied once per frame; the transformer processes 131 bands × 345 frames per 8-second chunk through 16 pairs of time and band layers.

This is a large model in the BS-RoFormer class. Its contribution is separation quality with standard public training data. The smaller 16–20M-parameter systems in the [leaderboards](results.md) occupy a different efficiency range. The default FP16 checkpoint is 533 MB; measured RTX 3090 throughput and end-to-end timings are in the [CLI guide](usage.md#compilation-and-measured-speed).

## Training and resources

Training uses only the **86 songs in the standard MUSDB18-HQ training split**, with 14 separate songs for validation. Eight-second training mixtures are formed by independently sampling and remixing the four stems, with random gains and stereo channel swaps. The objective combines waveform L1 loss with complex STFT L1 losses at five resolutions.

The training recipe uses AdamW, weight decay 0.01, global batch size 8, BF16, and an exponential moving average of the weights with decay 0.999. It starts with 400k steps on an 84-token plan, then refines the frequency partition through 40k + 100k steps on a 115-token plan and 60k steps on the final 131-token plan. Finer bands inherit slices of their parent bands' trained projections and mask heads. EMA checkpoint averaging produces the final single network; there is no ensemble to run at inference.

To fine-tune, save the [`.pt` checkpoint][weights-pt] under `weights/`, then run:

```bash
python -m gaborformer.train \
    --musdb /path/to/musdb18hq \
    --run-dir runs/finetune \
    --init-from weights/gaborformer.pt \
    --steps 60000 --lr 1e-4
```

The released `.pt` contains FP32 weights and configuration, but no optimizer state. It initializes a new training run; it is not a full training-state resume.

The trainer optimizes all four stems jointly; per-stem specialization requires adapting the loss. Additional tracks can be added under `train/` in the MUSDB18-HQ folder layout, preserving the standard 14 validation tracks. Training requires CUDA; `torchrun` supports multiple GPUs. Rerunning a training command with the same `--run-dir` resumes its local `last.pt`, including optimizer state.

Reproducing the released training recipe on **NVIDIA A100 or H100 Spot instances** is estimated to cost **a few thousand US dollars**, depending on provider, region, and interruptions.

### Keep experimenting

**If you want to keep training or fine-tuning, use the provided [`.pt` file][weights-pt]**—try extra data, per-stem specialization, or your own ideas.

## Acknowledgments

Gaborformer is released to support further music separation research and experimentation.

Thanks to the MUSDB18-HQ contributors and maintainers, the authors of NSGT and BS-RoFormer, and the open-source music separation community.

## References

- Sevag Hanssian. [Music demixing with the sliCQ transform][xumx-paper]. MDX Workshop, 2021.
- Peter Balazs, Monika Dörfler, Florent Jaillet, Nicki Holighaus, and Gino Velasco. [Theory, implementation and applications of nonstationary Gabor frames][nsgt-paper]. Journal of Computational and Applied Mathematics, 2011.
- Nicki Holighaus, Monika Dörfler, Gino Angelo Velasco, and Thomas Grill. [A framework for invertible, real-time constant-Q transforms][slicq-paper]. IEEE Transactions on Audio, Speech, and Language Processing, 2013.
- Wei-Tsung Lu, Ju-Chiang Wang, Qiuqiang Kong, and Yun-Ning Hung. [Music Source Separation with Band-Split RoPE Transformer][bsrf-published]. ICASSP, 2024.
- Phil Wang. [BS-RoFormer][lucidrains]. PyTorch implementation (software).

[weights-pt]: https://huggingface.co/sevagh/gaborformer/resolve/v1.0.0/gaborformer.pt
[bsrf]: https://arxiv.org/html/2309.02612
[xumx]: https://github.com/sevagh/xumx-sliCQ
[lucidrains]: https://github.com/lucidrains/BS-RoFormer
[nsgt-paper]: https://doi.org/10.1016/j.cam.2011.09.011
[xumx-paper]: https://arxiv.org/abs/2112.05509
[slicq-paper]: https://doi.org/10.1109/TASL.2012.2234114
[bsrf-published]: https://doi.org/10.1109/ICASSP48485.2024.10446843
