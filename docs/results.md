# Evaluation and leaderboards

[← README](../README.md)

Scores are **cSDR in dB, higher is better**, on the 50-track MUSDB18-HQ test set. Gaborformer uses BSSEval v4, taking the median over 1-second windows within each track, then the median over tracks for each stem. The overall score is the mean of the four stem scores. Per-track scores are included for the [default CLI](../tests/results/gaborformer-default.json) and the [original FP32-checkpoint evaluation](../tests/results/gaborformer.json).

Tables A and B show five entries per comparison, ordered by average cSDR. **Their competitor scores come from the cited public benchmark tables.** Training splits, evaluation implementations, and compute budgets differ across sources. Parameter counts describe the complete four-stem system unless qualified; `4 ×` denotes separate models for the four stems. Sources checked: November 2026.

Gaborformer's leaderboard scores were measured with FP32-stored weights, BF16 autocast, 8-second chunks, 50% overlap, and batch size 8. The default 533 MB FP16 download reproduces all five scores at the reported two-decimal precision; see [released-checkpoint validation](#released-checkpoint-validation).

## A. Trained on MUSDB18-HQ only

| System | Parameters | Training songs | Avg | Vocals | Drums | Bass | Other |
|---|---:|---|---:|---:|---:|---:|---:|
| **Gaborformer** | **266.6M** | **86** | **11.46** | **12.05** | **12.18** | **12.27** | **9.34** |
| [DiCoSe, BS-RoFormer + consistency refiner (T=1)][dicose] | 131M + 180M | 100‡ | 10.41 | 11.02 | 12.15 | 10.20 | 8.27 |
| [BS-RoFormer L=9][mel] | 82.8M reported§ | HQ train† | 10.02 | 11.02 | 9.66 | 11.58 | 7.80 |
| [Moises-Light][moises] | 4 × 5M | 86 | 9.96 | 10.92 | 10.93 | 10.04 | 7.95 |
| [SFC-Locoformer Medium][sfc] | 16.0M | 86 | 9.95 | 9.94 | 11.25 | 10.85 | 7.77 |

† The source states MUSDB18-HQ training data without specifying the held-out validation split. Gaborformer uses the standard **86 training / 14 validation** track split, with no extra songs, external pretraining, or pseudo-labeled data.

‡ DiCoSe reports MUSDB18-HQ training and uses a pretrained ZFTurbo backbone. Its HQ-only classification follows the [matching upstream model listing][msst]; the paper does not separately document the backbone's training data.

§ The L=9 source reports 82.8M parameters. A four-model system is inferred from its continuation of the original stem-specific training setup; the source does not explicitly report a combined parameter count.

Gaborformer leads this comparison on the average and on all four stems. The average margin over DiCoSe is **1.05 dB**; the **0.03 dB** drum margin should be read as essentially tied. Published averages are retained as reported, including small inconsistencies with the arithmetic mean of rounded stem scores.

## B. Systems trained with extra data

These systems are evaluated on the same test benchmark. Gaborformer is repeated in italics to show its position using only the 86 training songs.

| System | Parameters | Training data | Avg | Vocals | Drums | Bass | Other |
|---|---:|---|---:|---:|---:|---:|---:|
| [BS-RoFormer L=12, SDX'23 winner][bsrf] | 3 × 93.4M | HQ + 450 train / 50 validation | 11.99 | 12.72 | 12.91 | 13.32 | 9.01 |
| *Gaborformer* | *266.6M* | *86 HQ songs only* | *11.46* | *12.05* | *12.18* | *12.27* | *9.34* |
| [Moises-Light + MoisesDB][moises-extra] | 4 × 5M | 86 HQ + 235 MoisesDB | 10.31 | 11.10 | 11.10 | 10.77 | 8.28 |
| [SCNet-large + MoisesDB][scnet] | 41.2M | 100 HQ + 235 MoisesDB | 9.92 | 11.10 | 11.23 | 9.86 | 7.51 |
| [BS-Locoformer Large + MixIT][mixit] | Not reported | 86 HQ + ~890 h unlabeled FMA | 9.90 | 10.39 | 10.96 | 10.21 | 8.06 |

The SDX'23 system uses three stem-specific models and obtains “other” by subtraction. The Moises-Light + MoisesDB average is calculated from its four published stem scores.

Whole-track SDR and cSDR are different metrics. For example, [SCNet XL IHF's 10.08 whole-track SDR][msst] is not ranked against cSDR here.

## Head-to-head with BS-RoFormer-SW

**We evaluated both checkpoints ourselves** on the same 50 MUSDB18-HQ test tracks using the cSDR aggregation described above. These SW scores are our measurements, not scores taken from a public leaderboard. Per-track results: [Gaborformer](../tests/results/gaborformer.json) · [BS-RoFormer-SW](../tests/results/bs-roformer-sw.json).

| System | Avg | Vocals | Drums | Bass | Other |
|---|---:|---:|---:|---:|---:|
| BS-RoFormer-SW | **11.80** | 11.98 | **13.28** | 12.24 | **9.72** |
| Gaborformer | 11.46 | **12.05** | 12.18 | **12.27** | 9.34 |

SW leads by **0.34 dB** on average; Gaborformer scores slightly higher on vocals and bass. SW uses FP16 inference, 588,800-sample chunks and 50% overlap, with guitar and piano summed into “other.” Gaborformer uses the settings stated above. This compares released systems, not architectures under matched training conditions.

## Released-checkpoint validation

Each row below is a separate evaluation of all **50 MUSDB18-HQ test tracks** through the public inference implementation using the **released FP16 safetensors checkpoint**. Weights are loaded into an FP32 model; the precision option selects the computation dtype. All runs use 8-second chunks, 50% overlap, and batch size 8. Row links contain the full per-track results.

| Setting | Avg | Vocals | Drums | Bass | Other |
|---|---:|---:|---:|---:|---:|
| [Default: BF16](../tests/results/gaborformer-default.json) | **11.46** | 12.05 | 12.18 | 12.27 | 9.34 |
| [BF16 with `--compile`](../tests/results/gaborformer-compile.json) | **11.46** | 12.05 | 12.18 | 12.27 | 9.34 |
| [`--precision fp16`](../tests/results/gaborformer-fp16.json) | **11.46** | 12.06 | 12.18 | 12.26 | 9.36 |
| [`--precision fp32`](../tests/results/gaborformer-fp32.json) | **11.46** | 12.06 | 12.18 | 12.26 | 9.36 |

All four configurations give the same rounded average cSDR. Paired against the default, each stem's mean cSDR difference is below **0.007 dB** in magnitude. These are mean differences across tracks; individual outputs differ. Checkpoint names and configuration metadata in the JSONs use the public release format; evaluation scores are preserved unchanged.

See the [CLI guide](usage.md#compilation-and-measured-speed) for measured inference times and the cold/warm compilation trade-off.

[dicose]: https://arxiv.org/html/2412.06965v2#S5.T2.4.1
[bsrf]: https://arxiv.org/html/2309.02612
[mel]: https://arxiv.org/html/2310.01809#S1.T1.2
[moises]: https://arxiv.org/html/2510.06785v1#S5.T4.2.1
[moises-extra]: https://arxiv.org/html/2510.06785v1#S5.T5
[sfc]: https://arxiv.org/html/2602.08671#S5.T1.9.1
[scnet]: https://arxiv.org/html/2401.13276#S4.T2.1
[mixit]: https://arxiv.org/html/2505.07631v1#S4.T1.2.1
[msst]: https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/main/docs/pretrained_models.md
