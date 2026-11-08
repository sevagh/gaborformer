# Pretrained weights

Download the checkpoints from [Hugging Face](https://huggingface.co/sevagh/gaborformer) or the R2 mirror:

| File | Size | Hugging Face | R2 mirror |
|---|---:|---|---|
| `gaborformer-fp16.safetensors` | 533 MB | [Download](https://huggingface.co/sevagh/gaborformer/resolve/v1.0.0/gaborformer-fp16.safetensors) | [Download](https://gaborformer.sevag.xyz/gaborformer-fp16.safetensors) |
| `gaborformer-fp32.safetensors` | 1.07 GB | [Download](https://huggingface.co/sevagh/gaborformer/resolve/v1.0.0/gaborformer-fp32.safetensors) | [Download](https://gaborformer.sevag.xyz/gaborformer-fp32.safetensors) |
| `gaborformer.pt` | 1.07 GB | [Download](https://huggingface.co/sevagh/gaborformer/resolve/v1.0.0/gaborformer.pt) | [Download](https://gaborformer.sevag.xyz/gaborformer.pt) |

Use FP16 for inference, or FP32 / `.pt` for further training. Both hosts provide the same files; verify downloads against [SHA256SUMS](SHA256SUMS). The weights use the [non-commercial license](LICENSE).

The CLI downloads from Hugging Face by default. To use a manually downloaded checkpoint:

```bash
gaborformer song.wav --weights weights/gaborformer-fp16.safetensors
```
