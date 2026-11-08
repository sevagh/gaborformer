# Gaborformer

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-sevagh%2Fgaborformer-yellow)](https://huggingface.co/sevagh/gaborformer)

Gaborformer is a **music stem separation model** released by [sevagh](https://github.com/sevagh) (Sevag Hanssian).

:robot: Gaborformer was developed with substantial AI coding assistance; see [AI assistance](#ai-assistance).

## Abstract

Gaborformer revisits NSGT-based spectral masking for music source separation, a revisit of my older project [xumx-sliCQ](https://github.com/sevagh/xumx-sliCQ) ([Hanssian, 2021][xumx-paper]). It combines an invertible, multi-resolution **nonstationary Gabor transform** ([Balazs et al., 2011][nsgt-paper]; [Holighaus et al., 2013][slicq-paper]) with a **BS-RoFormer-style network** ([Lu et al., 2024][bsrf-paper]), building on [Phil Wang's PyTorch implementation][lucidrains]. Trained exclusively on the standard **86-track training subset of MUSDB18-HQ**, a single model separates **vocals, drums, bass, and other** and achieves **11.46 dB average cSDR** on the **50-track test set**.

The model has **266.6M parameters** and **533 MB** FP16 weights. Inference code, training code, and per-track evaluation results are included.

## Quickstart

Requires Python 3.10+ and PyTorch 2.4+. Use a CUDA-enabled PyTorch build for GPU inference.

```bash
git clone https://github.com/sevagh/gaborformer.git
cd gaborformer
python -m pip install -e .
gaborformer song.wav --out separated
```

Weights download automatically from [Hugging Face](https://huggingface.co/sevagh/gaborformer) and are cached locally. The CLI creates `separated/song/` with four stem WAVs. Inputs are automatically resampled to 44.1 kHz; M4A/AAC requires system FFmpeg, included in Docker.

- [CLI guide](docs/usage.md): uv, audio formats, local weights, offline use, and options.
- [Docker guide](docs/docker.md): NVIDIA GPU setup and a three-way listening comparison with DiCoSe and BS-RoFormer-SW.

## Results

Gaborformer has the highest average cSDR in the [MUSDB18-HQ-only comparison](docs/results.md#a-trained-on-musdb18-hq-only): **+1.05 dB over DiCoSe** and **+1.44 dB over BS-RoFormer L=9**. It is **0.53 dB below the SDX'23-winning BS-RoFormer**, which used 450 additional training songs and 50 additional validation songs.

Scores are **cSDR in dB (higher is better)** on the 50-track MUSDB18-HQ test set. Competitor scores come from the cited public benchmark tables; training splits and evaluation implementations differ.

Parameter counts describe the complete system unless noted; `×` denotes separate stem models.

### A. Trained on MUSDB18-HQ only

| System | Parameters | Training songs | Avg | Vocals | Drums | Bass | Other |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Gaborformer** | **266.6M** | **86** | **11.46** | **12.05** | **12.18** | **12.27** | **9.34** |
| [DiCoSe (T=1)][dicose] | 131M + 180M | 100‡ | 10.41 | 11.02 | 12.15 | 10.20 | 8.27 |
| [BS-RoFormer L=9][mel] | 82.8M reported§ | HQ train† | 10.02 | 11.02 | 9.66 | 11.58 | 7.80 |
| [Moises-Light][moises] | 4 × 5M | 86 | 9.96 | 10.92 | 10.93 | 10.04 | 7.95 |
| [SFC-Locoformer Medium][sfc] | 16.0M | 86 | 9.95 | 9.94 | 11.25 | 10.85 | 7.77 |

† The source does not specify the held-out validation split.

‡ DiCoSe's backbone training data are inferred from its upstream model listing; see [provenance details](docs/results.md#a-trained-on-musdb18-hq-only).

§ The L=9 source reports 82.8M parameters without an explicit combined four-stem system count.

### B. Systems trained with extra data

| System | Parameters | Training data | Avg | Vocals | Drums | Bass | Other |
|---|---:|---|---:|---:|---:|---:|---:|
| [BS-RoFormer L=12, SDX'23 winner][bsrf] | 3 × 93.4M | HQ + 450 train / 50 validation | 11.99 | 12.72 | 12.91 | 13.32 | 9.01 |
| *Gaborformer* | *266.6M* | *86 HQ songs only* | *11.46* | *12.05* | *12.18* | *12.27* | *9.34* |
| [Moises-Light + MoisesDB][moises-extra] | 4 × 5M | 86 HQ + 235 MoisesDB | 10.31 | 11.10 | 11.10 | 10.77 | 8.28 |
| [SCNet-large + MoisesDB][scnet] | 41.2M | 100 HQ + 235 MoisesDB | 9.92 | 11.10 | 11.23 | 9.86 | 7.51 |
| [BS-Locoformer Large + MixIT][mixit] | Not reported | 86 HQ + ~890 h unlabeled FMA | 9.90 | 10.39 | 10.96 | 10.21 | 8.06 |

[Full results](docs/results.md) include evaluation settings, source details, and our measured head-to-head with BS-RoFormer-SW.

## Keep experimenting

**If you want to keep training or fine-tuning, use the provided [`.pt` file](https://huggingface.co/sevagh/gaborformer/resolve/v1.0.0/gaborformer.pt).**

- [Pretrained weights](weights/README.md): Hugging Face downloads, R2 mirror, and checksums.
- [Architecture and training](docs/research.md): NSGT design, training recipe, fine-tuning, and compute budget.
- [Development](docs/development.md): tests and reference-output verification.

## AI assistance

This project was developed with substantial AI coding assistance, including Anthropic Claude, OpenAI Codex, and Google Gemini. It was driven by the author's longstanding interest in bringing the nonstationary Gabor transform to music source separation, continuing the idea explored in the 2021 master's thesis project [xumx-sliCQ](https://github.com/sevagh/xumx-sliCQ). The author directed the research and is responsible for the released code and reported results.

## License

Code: [MIT](LICENSE). Pretrained weights: [non-commercial use only](weights/LICENSE), trained exclusively on [MUSDB18-HQ](https://sigsep.github.io/datasets/musdb.html).

[dicose]: https://arxiv.org/html/2412.06965v2#S5.T2.4.1
[bsrf]: https://arxiv.org/html/2309.02612
[mel]: https://arxiv.org/html/2310.01809#S1.T1.2
[moises]: https://arxiv.org/html/2510.06785v1#S5.T4.2.1
[moises-extra]: https://arxiv.org/html/2510.06785v1#S5.T5
[sfc]: https://arxiv.org/html/2602.08671#S5.T1.9.1
[scnet]: https://arxiv.org/html/2401.13276#S4.T2.1
[mixit]: https://arxiv.org/html/2505.07631v1#S4.T1.2.1
[xumx-paper]: https://arxiv.org/abs/2112.05509
[nsgt-paper]: https://doi.org/10.1016/j.cam.2011.09.011
[slicq-paper]: https://doi.org/10.1109/TASL.2012.2234114
[bsrf-paper]: https://doi.org/10.1109/ICASSP48485.2024.10446843
[lucidrains]: https://github.com/lucidrains/BS-RoFormer
