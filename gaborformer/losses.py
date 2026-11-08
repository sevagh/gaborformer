"""Waveform L1 + multi-resolution STFT L1 (L1 on the real and imaginary parts of the STFT coefficients).

Window sizes follow lucidrains/BS-RoFormer; each resolution uses n_fft = window and hop = window / 4.
"""
import torch
import torch.nn.functional as F

MR_WINDOWS = (4096, 2048, 1024, 512, 256)


def mrstft_l1(est, ref, windows=MR_WINDOWS):
    est, ref = est.reshape(-1, est.shape[-1]), ref.reshape(-1, ref.shape[-1])
    loss = 0.0
    for w in windows:
        kw = dict(n_fft=w, hop_length=w // 4, win_length=w, window=torch.hann_window(w, device=est.device), return_complex=True)
        loss = loss + F.l1_loss(torch.view_as_real(torch.stft(est, **kw)), torch.view_as_real(torch.stft(ref, **kw)))
    return loss


def separation_loss(est, ref, mrstft_weight=1.0):
    """est, ref: (batch, stems, channels, n). Returns total, (waveform L1, MR-STFT L1)."""
    est, ref = est.float(), ref.float()
    l1 = F.l1_loss(est, ref)
    mr = mrstft_l1(est, ref)
    return l1 + mrstft_weight * mr, (l1, mr)
