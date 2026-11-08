"""MUSDB18-HQ: random-mix training chunks and full-track loading.

A dataset root holds train/ and test/ folders of track folders, each with mixture.wav, vocals.wav, drums.wav,
bass.wav and other.wav at 44.1 kHz stereo (the MUSDB18-HQ layout). The 14 standard validation tracks are held out of
the training split.
"""
import os

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .model import STEMS

# musdb 0.4.3 configs/mus.yaml: the standard 14-track validation split of MUSDB18 train
VALIDATION = (
    "Actions - One Minute Smile",
    "Clara Berry And Wooldog - Waltz For My Victims",
    "Johnny Lokke - Promises & Lies",
    "Patrick Talbot - A Reason To Leave",
    "Triviul - Angelsaint",
    "Alexander Ross - Goodbye Bolero",
    "Fergessen - Nos Palpitants",
    "Leaf - Summerghost",
    "Skelpolu - Human Mistakes",
    "Young Griffo - Pennies",
    "ANiMAL - Rockshow",
    "James May - On The Line",
    "Meaxic - Take A Step",
    "Traffic Experiment - Sirens",
)


def musdb_root(path=None):
    root = path or os.environ.get("MUSDB_PATH")
    if not root or not os.path.isdir(os.path.join(root, "train")):
        raise SystemExit("point --musdb (or MUSDB_PATH) at a MUSDB18-HQ folder holding train/ and test/")
    return root


def list_tracks(root, subset):
    """subset: 'train' (86 tracks), 'valid' (14), or 'test' (50)."""
    split = "test" if subset == "test" else "train"
    d = os.path.join(root, split)
    names = sorted(t for t in os.listdir(d) if os.path.isfile(os.path.join(d, t, "mixture.wav")))
    if subset == "train":
        names = [t for t in names if t not in VALIDATION]
    elif subset == "valid":
        names = [t for t in names if t in VALIDATION]
        assert len(names) == len(VALIDATION), "validation tracks missing from MUSDB18-HQ train"
    return [os.path.join(d, t) for t in names]


def load_track(path):
    """-> mixture (2, n), stems (4, 2, n) float32 tensors."""
    read = lambda name: sf.read(os.path.join(path, name + ".wav"), dtype="float32", always_2d=True)[0].T
    stems = np.stack([read(s) for s in STEMS])
    return torch.from_numpy(read("mixture").copy()), torch.from_numpy(stems)


class RandomMix(Dataset):
    """Each item mixes one random chunk per stem, each from an independently chosen track, with a uniform gain in
    [gain_lo, gain_hi] and a random left/right swap per stem. Randomness comes from torch's per-worker seeded
    generator, so DataLoader workers differ."""

    def __init__(self, root, length=352800, samples_per_epoch=8000, gain=(0.25, 1.25), subset="train"):
        self.tracks = list_tracks(root, subset)
        self.length, self.samples, self.gain = length, samples_per_epoch, gain
        self.frames = [sf.info(os.path.join(t, "mixture.wav")).frames for t in self.tracks]

    def __len__(self):
        return self.samples

    def _chunk(self, stem):
        t = int(torch.randint(len(self.tracks), ()))
        start = int(torch.randint(max(1, self.frames[t] - self.length), ()))
        x, _ = sf.read(os.path.join(self.tracks[t], stem + ".wav"), start=start, frames=self.length, dtype="float32", always_2d=True)
        x = x.T
        if x.shape[-1] < self.length:
            x = np.pad(x, ((0, 0), (0, self.length - x.shape[-1])))
        x = torch.from_numpy(np.ascontiguousarray(x))
        g = self.gain[0] + (self.gain[1] - self.gain[0]) * float(torch.rand(()))
        if float(torch.rand(())) < 0.5:
            x = x.flip(0)
        return x * g

    def __getitem__(self, i):
        stems = torch.stack([self._chunk(s) for s in STEMS])  # (4, 2, length)
        return stems.sum(0), stems
