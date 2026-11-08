"""Pinned public checkpoints and inference adapters for the Docker listening tool.

Upstream implementations are installed in the image, not vendored into this repo.
These adapters are for listening comparisons, not reproducing benchmark scores.
"""
from argparse import Namespace
from pathlib import Path
import pickle
import sys
from types import ModuleType

ARTIFACTS = {
    "dicose": {
        "code": "https://github.com/Russell-Izadi-Bose/DiCoSe",
        "commit": "a1dc0a41ad2b1829674a60ea74e74edfd1509083",
        "repo_id": "karchkha/DiCoSe",
        "revision": "b3e44147b96e55b08eea2dd0b6b4e017748a87a9",
        "files": {
            "Deterministic_model_MSST_bs_roformer/model.ckpt": "8087fbdcbc63f11f3ee305ef042cf42a42a5802e8a76678997f6448cb45256f5",
            "CD_MSST_bs_roformer/model.ckpt": "d25035bed7294a227fcb0f1ea691a0d1b8452ef76bde0e411c2b75536acf13da",
        },
    },
    "bs-roformer-sw": {
        "code": "https://github.com/openmirlab/bs-roformer-infer",
        "commit": "95c8de7b0b091d00f064d694bf916a21866c32f5",
        "repo_id": "enerjazzer/BS-ROFO-SW-Fixed",
        "revision": "a443a2985534b3bc815ef54a5d446c6a0390f974",
        "files": {
            "BS-Rofo-SW-Fixed.ckpt": "24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e",
            "BS-Rofo-SW-Fixed.yaml": "f9fada9f94e5ba2d2e4600196299459294bc5f532b314c209cc156ac63e4329b",
        },
    },
}


def download(name, offline=False):
    from huggingface_hub import hf_hub_download
    from gaborformer.pretrained import sha256

    spec = ARTIFACTS[name]
    paths = {}
    for filename, digest in spec["files"].items():
        path = hf_hub_download(spec["repo_id"], filename, revision=spec["revision"],
                               local_files_only=offline)
        if sha256(path) != digest:
            raise RuntimeError(f"SHA-256 mismatch: {filename}; refusing to load checkpoint")
        paths[filename] = path
    return paths


def namespace(value):
    if isinstance(value, dict):
        return Namespace(**{key: namespace(item) for key, item in value.items()})
    return value


class _TrainingDatasetReference:
    """Placeholder for unused dataset constructors in checkpoint hyperparameters."""


class _InferenceUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in {
            ("main.data", "MUSDB_Aug_Repitch_Tracks_Dataset_MSST"),
            ("main.msst.utils.dataset", "MSSDataset"),
            ("main.musdb_aug_repitch_tools.wav", "Wavset"),
        }:
            return _TrainingDatasetReference
        return super().find_class(module, name)


def read_dicose_checkpoint(path):
    """Read hash-verified upstream weights without importing training datasets."""
    import torch

    loader = ModuleType("inference_pickle")
    loader.Unpickler = _InferenceUnpickler
    loader.load = pickle.load
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False, pickle_module=loader)


def load_dicose(paths, device, source=Path("/opt/dicose")):
    import torch
    import yaml

    sys.path[:0] = [str(source), str(source / "ctm")]
    from ctm.utils import create_model_and_diffusion_audio_msst
    from ctm.sample_util import karras_sample
    from main.msst.models.bs_roformer import BSRoformer

    config = yaml.load((source / "configs/consistency_model/bsrf_eval.yaml").read_text(),
                       Loader=yaml.FullLoader)
    config["model"].pop("_target_", None)  # Hydra's Lightning wrapper, not a network argument.
    cfg = namespace(config)
    det_cfg = vars(cfg.model.pre_trained_mixture_feature_extractor_model_config).copy()
    for key in ("_target_", "learning_rate", "patience", "reduce_factor", "model_type"):
        det_cfg.pop(key, None)
    deterministic = BSRoformer(**det_cfg)
    student, diffusion = create_model_and_diffusion_audio_msst(cfg)

    # Load only inference networks from the authors' Lightning checkpoints.
    # mmap avoids materializing unused optimizer/teacher/EMA tensors in RAM.
    for model, filename, prefix in (
        (deterministic, "Deterministic_model_MSST_bs_roformer/model.ckpt", "model.unet."),
        (student, "CD_MSST_bs_roformer/model.ckpt", "net."),
    ):
        checkpoint = read_dicose_checkpoint(paths[filename])
        state = {key[len(prefix):]: value for key, value in checkpoint["state_dict"].items()
                 if key.startswith(prefix)}
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        del state, checkpoint

    @torch.inference_mode()
    def predict(chunk):
        chunk = chunk.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            features = deterministic.get_feature(chunk)
            result = karras_sample(
                diffusion, student, shape=(chunk.shape[0], 4, 2, chunk.shape[-1]),
                steps=1, model_kwargs={"features": None, "channels_list": None,
                                      "embedding": None, "mixture_features_channels_list": list(features)},
                device=device, clip_denoised=False, sampler="cm_multistep_cd",
                teacher=False, ctm=False, clip_output=True,
                sigma_min=cfg.diffusion.sigma_min, sigma_max=0.003934,
                rho=cfg.diffusion.rho,
            )
        return result.float().cpu()

    return predict


def dicose(paths, mix, device):
    from gaborformer.separate import separate

    predict = load_dicose(paths, device)

    class ChunkModel:
        length = 485100
        n_stems = 4

        def __call__(self, chunk):
            return predict(chunk)

    # Eleven-second chunks, 50% triangular overlap; accumulation stays on CPU.
    output = separate(ChunkModel(), mix, batch_size=1, precision="fp32")
    return dict(zip(("drums", "bass", "other", "vocals"), output.numpy()))


def sw_four_stems(stems):
    return {"vocals": stems["vocals"], "drums": stems["drums"], "bass": stems["bass"],
            "other": stems["other"] + stems["guitar"] + stems["piano"]}


def load_sw(paths, device):
    import yaml
    from ml_collections import ConfigDict
    from bs_roformer.inference import SafeLoaderWithTuple
    from bs_roformer.utils import get_model_from_config, load_checkpoint_state

    config = ConfigDict(yaml.load(Path(paths["BS-Rofo-SW-Fixed.yaml"]).read_text(),
                                  Loader=SafeLoaderWithTuple))
    model = get_model_from_config("bs_roformer", config)
    model.load_state_dict(load_checkpoint_state(paths["BS-Rofo-SW-Fixed.ckpt"]), strict=True)
    return model.to(device).eval(), config


def bs_roformer_sw(paths, mix, device):
    from bs_roformer.utils import demix_track

    model, config = load_sw(paths, device)
    stems, _ = demix_track(config, model, mix, device)
    return sw_four_stems(stems)
