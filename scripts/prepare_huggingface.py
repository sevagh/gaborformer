"""Prepare a local Hugging Face upload folder; does not create a repository or upload anything.

    python -m scripts.prepare_huggingface
    python -m scripts.prepare_huggingface --card-only

The default output, .release/huggingface/, is ignored by Git. Its model card is generated from the
root README so that results, credits, and disclosures have one maintained source. Checkpoints are
verified and hard-linked when possible, otherwise copied. Original files are never moved or removed.
Use --card-only to generate just README.md without checkpoints or third-party dependencies.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import struct

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/sevagh/gaborformer"


def model_card(readme):
    metadata = """---
license: other
license_name: gaborformer-noncommercial
license_link: LICENSE
pipeline_tag: audio-to-audio
tags:
  - pytorch
  - music-source-separation
  - nsgt
  - musdb18-hq
---

"""
    readme = re.sub(r"^> \*\*Release staging:\*\*.*\n\n", "", readme, flags=re.MULTILINE)
    readme = readme.replace("This GitHub repository", "The GitHub code repository")
    readme = readme.replace("From a clone of this repository:", f"From a clone of the [GitHub code repository]({GITHUB}):")
    readme = readme.replace("(tests/results/", f"({GITHUB}/blob/main/tests/results/")
    readme = readme.replace("(docs/", f"({GITHUB}/blob/main/docs/")
    readme = readme.replace("(weights/README.md)", f"({GITHUB}/blob/main/weights/README.md)")
    readme = readme.replace("(LICENSE)", f"({GITHUB}/blob/main/LICENSE)")
    readme = readme.replace("(weights/LICENSE)", "(LICENSE)")
    readme = readme.replace("(weights/SHA256SUMS)", "(SHA256SUMS)")
    return metadata + readme


def prepare(out_dir):
    from gaborformer.pretrained import WEIGHTS, sha256

    out_dir = Path(out_dir)
    sources = [(ROOT / "weights" / artifact["filename"], artifact) for artifact in WEIGHTS.values()]
    # Check all inputs before writing the bundle; hashing streams through a small buffer.
    for source, artifact in sources:
        if sha256(source) != artifact["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {source}")
    sums = "".join(f"{a['sha256']}  {a['filename']}\n" for a in WEIGHTS.values())
    if (ROOT / "weights/SHA256SUMS").read_text() != sums:
        raise ValueError("weights/SHA256SUMS does not match the release manifest")
    with sources[0][0].open("rb") as f:
        header_length = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_length))
    config = {
        "model_type": "gaborformer",
        "architectures": ["Gaborformer"],
        "sample_rate": 44100,
        "sources": ["vocals", "drums", "bass", "other"],
        "model_config": json.loads(header["__metadata__"]["cfg"]),
        "weights": WEIGHTS,
    }
    # Refuse an existing folder to avoid uploading stale or unexpected files alongside the release.
    out_dir.mkdir(parents=True, exist_ok=False)
    for source, artifact in sources:
        destination = out_dir / artifact["filename"]
        try:
            os.link(source, destination)
        except OSError:
            shutil.copyfile(source, destination)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out_dir / "SHA256SUMS").write_text(sums)
    shutil.copyfile(ROOT / "weights/LICENSE", out_dir / "LICENSE")
    (out_dir / "README.md").write_text(model_card((ROOT / "README.md").read_text()))
    return out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=ROOT / ".release/huggingface")
    p.add_argument("--card-only", action="store_true", help="Generate only README.md; no checkpoints required")
    a = p.parse_args()
    try:
        if a.card_only:
            a.out_dir.mkdir(parents=True, exist_ok=True)
            destination = a.out_dir / "README.md"
            destination.write_text(model_card((ROOT / "README.md").read_text()))
        else:
            destination = prepare(a.out_dir)
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    if a.card_only:
        print(f"Prepared model card: {destination}")
    else:
        print(f"Prepared {destination}: three checkpoints, config.json, SHA256SUMS, LICENSE, README.md")
    print("No remote repository was created and no files were uploaded.")


if __name__ == "__main__":
    main()
