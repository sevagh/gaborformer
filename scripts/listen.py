"""Docker entry point: separate a track with independently loaded models."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

MODELS = ("gaborformer", "dicose", "bs-roformer-sw")
STEMS = ("vocals", "drums", "bass", "other")


def read_audio(path):
    from gaborformer.audio import read_audio as decode

    return decode(path)


def write_stems(destination, stems, samples):
    import numpy as np
    import soundfile as sf

    # Validate all outputs before replacing any previous result.
    for name in STEMS:
        value = stems[name]
        if value.shape != (2, samples) or not np.isfinite(value).all():
            raise RuntimeError(f"Invalid {name} output: expected finite stereo audio of length {samples}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in STEMS:
        sf.write(destination / f"{name}.wav", stems[name].T, 44100, subtype="FLOAT")


def worker(args):
    import torch
    from gaborformer import pretrained
    from scripts import competitors

    name = args.worker
    if name == "gaborformer":
        path = Path(args.weights) if args.weights else pretrained.download_weights(offline=args.offline)
        metadata = {"checkpoint_sha256": pretrained.sha256(path)}
        if not args.weights:
            metadata.update(repo_id=pretrained.REPO_ID, revision=pretrained.REVISION)
    else:
        paths = competitors.download(name, offline=args.offline)
        metadata = competitors.ARTIFACTS[name].copy()
    if args.download_only:
        print(f"{name}: checkpoint download and checksum verification complete", flush=True)
        return

    import soundfile as sf
    audio, rate = sf.read(args.audio, dtype="float32", always_2d=True)
    assert rate == 44100 and audio.shape[1] == 2
    mix = torch.from_numpy(audio.T.copy())
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --gpus all and check the README's NVIDIA device notes, "
                           "or explicitly select --cpu.")
    device = torch.device("cpu" if args.cpu else "cuda")
    torch.manual_seed(args.seed)
    print(f"Separating with {name} on {device}", flush=True)
    if name == "gaborformer":
        from gaborformer.separate import load_model, separate
        model, _ = load_model(path, device)
        result = separate(model, mix.to(device), batch_size=args.batch_size, precision=args.precision)
        stems = dict(zip(STEMS, result.cpu().numpy()))
        metadata.update(precision=args.precision if device.type == "cuda" else "fp32",
                        chunk_samples=model.length, overlap=0.5, batch_size=args.batch_size)
    elif name == "dicose":
        stems = competitors.dicose(paths, mix, device)
        metadata.update(chunk_samples=485100, overlap=0.5, sampler="cm_multistep_cd",
                        steps=1, sigma_max=0.003934, clip_output=True, seed=args.seed)
    else:
        stems = competitors.bs_roformer_sw(paths, mix, device)
        metadata.update(stem_mapping={"other": ["other", "guitar", "piano"]})
    metadata["device"] = str(device)
    destination = Path(args.out)
    write_stems(destination, stems, len(audio))
    (destination / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")


def run_models(args, audio=None, destination=None):
    selected = MODELS if args.include_competitors else MODELS[:1]
    for name in selected:
        command = [sys.executable, "-m", "scripts.listen", "--worker", name,
                   "--precision", args.precision, "--batch-size", str(args.batch_size),
                   "--seed", str(args.seed)]
        if audio is not None:
            command += [str(audio), "--out", str(destination / name)]
        for option in ("cpu", "offline", "download_only"):
            if getattr(args, option):
                command.append("--" + option.replace("_", "-"))
        if args.weights and name == "gaborformer":
            command += ["--weights", str(Path(args.weights).resolve())]
        # A separate process frees each model and its CUDA context before the next.
        subprocess.run(command, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Separate a track; optionally compare DiCoSe and BS-RoFormer-SW.")
    parser.add_argument("audio", nargs="?", help="input audio path inside the container")
    parser.add_argument("--out", default="separated", help="output root (default: separated)")
    parser.add_argument("--include-competitors", action="store_true", help="run all three models sequentially")
    parser.add_argument("--download-only", action="store_true", help="cache and verify weights without inference; no audio needed")
    parser.add_argument("--offline", action="store_true", help="require already-cached checkpoints")
    parser.add_argument("--weights", help="local Gaborformer checkpoint")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16", help="Gaborformer CUDA precision")
    parser.add_argument("--batch-size", type=int, default=1, help="Gaborformer batch size (default: 1)")
    parser.add_argument("--seed", type=int, default=0, help="DiCoSe sampling seed (default: 0)")
    parser.add_argument("--worker", choices=MODELS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.audio and not args.download_only:
        parser.error("an audio file is required unless --download-only is used")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.weights and not Path(args.weights).is_file():
        parser.error(f"checkpoint not found: {args.weights}")
    try:
        if args.worker:
            worker(args)
        elif args.download_only:
            run_models(args)
        else:
            import soundfile as sf
            from gaborformer.pretrained import sha256

            source = Path(args.audio).resolve()
            audio = read_audio(source)
            destination = Path(args.out).resolve() / source.stem
            destination.mkdir(parents=True, exist_ok=True)
            manifest = {"input": source.name, "input_sha256": sha256(source), "sample_rate": 44100,
                        "samples": len(audio), "channels": 2, "subtype": "FLOAT",
                        "models": list(MODELS if args.include_competitors else MODELS[:1]),
                        "status": "running"}
            manifest_path = destination / "comparison.json"
            if manifest_path.exists():
                previous = json.loads(manifest_path.read_text())
                if previous.get("input_sha256") != manifest["input_sha256"]:
                    raise ValueError("Output directory contains a different input; choose another --out directory")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            try:
                # All models receive exactly the same resampled, unnormalized audio.
                # Temporary outputs replace previous files only after all models succeed.
                with tempfile.TemporaryDirectory(prefix=".listen-", dir=destination) as temp:
                    temp = Path(temp)
                    prepared = temp / "input.wav"
                    sf.write(prepared, audio, 44100, subtype="FLOAT")
                    run_models(args, prepared, temp)
                    for name in manifest["models"]:
                        (destination / name).mkdir(exist_ok=True)
                        for output in (temp / name).iterdir():
                            output.replace(destination / name / output.name)
            except Exception:
                manifest["status"] = "failed"
                raise
            else:
                manifest["status"] = "complete"
                print(f"Stems written to {destination}", flush=True)
            finally:
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Separation failed: {exc}\n")


if __name__ == "__main__":
    main()
