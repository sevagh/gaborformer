"""Download and verify the released checkpoints from Hugging Face."""
import hashlib
import json
from pathlib import Path

REPO_ID = "sevagh/gaborformer"
# Pin downloads to the immutable v1.0.0 model commit.
REVISION = "d0fccf6e32214d5e8a5a1d0501d0d53a2f82d85c"
WEIGHTS = {
    "fp16": {
        "filename": "gaborformer-fp16.safetensors",
        "sha256": "cd586fa45a551adf4d1fc3025cd415b16917e9de5e603e379bf9b10e5be88c4a",
    },
    "fp32": {
        "filename": "gaborformer-fp32.safetensors",
        "sha256": "f2e905f4fe927bff1adb24118d67efffdfdf628baa61c3fbdfb418f291d367e1",
    },
    "pt": {
        "filename": "gaborformer.pt",
        "sha256": "56773c619854e6edc29691270cdea7cd3bc6c7ae26c36f1655e8268a5ada72bb",
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download_weights(variant="fp16", *, offline=False, revision=None):
    """Return a verified cached checkpoint, using the normal HF login for private repos.

    `offline=True` only reads the cache. An explicit revision overrides the pinned commit
    but must still match this release's checkpoint manifest and checksums.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
    from httpx import HTTPError

    artifact = WEIGHTS[variant]
    revision = revision or REVISION
    kwargs = {"repo_id": REPO_ID, "revision": revision, "local_files_only": offline}
    try:
        config_path = hf_hub_download(filename="config.json", **kwargs)
        with open(config_path) as f:
            config = json.load(f)
        if (not isinstance(config, dict) or config.get("model_type") != "gaborformer"
                or not isinstance(config.get("weights"), dict) or config["weights"].get(variant) != artifact):
            raise RuntimeError("The downloaded config.json does not match this Gaborformer release.")
        path = Path(hf_hub_download(filename=artifact["filename"], **kwargs))
    except LocalEntryNotFoundError as exc:
        raise RuntimeError(
            "The requested Gaborformer release is not available in the local Hugging Face cache. "
            "Download it online first, or use --weights /path/to/checkpoint."
        ) from exc
    except (HfHubHTTPError, HTTPError) as exc:
        raise RuntimeError(
            f"Could not download {REPO_ID} at revision {revision}. "
            "Check the repository, revision, and your connection; private access requires `hf auth login`. "
            "Alternatively, use --weights /path/to/checkpoint."
        ) from exc
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"Could not read the cached Gaborformer release: {exc}") from exc

    if sha256(path) != artifact["sha256"]:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}. The checkpoint will not be loaded. "
            "Remove this cached file and download it again, or use --weights with a verified local file."
        )
    return path
