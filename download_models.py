"""
STEP 1 - run this ONCE, with internet ON.  (Everything after this works offline.)

Downloads into this folder:
  models/whisper-<size>   speech -> text, finds the language by itself
  models/nllb-600m        text -> text, ~200 languages (or nllb-1.3b: better, bigger)
  vendor/                 two small files the web page needs (styles and icons)

  python download_models.py                         # large-v3 + nllb 600M
  python download_models.py --translator 1.3b       # better translation quality
  python download_models.py --whisper medium        # smaller / faster speech model

NOTE: the NLLB translation models are licensed CC-BY-NC 4.0 (non-commercial use).
"""
import argparse
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

BASE = Path(__file__).resolve().parent
MODELS_DIR = BASE / "models"
VENDOR_DIR = BASE / "vendor"

WHISPER_REPOS = {
    "large-v3": "Systran/faster-whisper-large-v3",   # ~3 GB
    "medium": "Systran/faster-whisper-medium",       # ~1.5 GB
    "small": "Systran/faster-whisper-small",         # ~0.5 GB
}
# folder names must match TRANSLATOR_ORDER in pipeline.py
TRANSLATOR_REPOS = {
    "600m": ("facebook/nllb-200-distilled-600M", "nllb-600m"),    # ~2.5 GB
    "1.3b": ("facebook/nllb-200-distilled-1.3B", "nllb-1.3b"),    # ~5.5 GB
}
PLAYER_FILES = [
    ("https://cdn.tailwindcss.com/3.4.17", "tailwind.js"),
    ("https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/js/all.min.js", "fontawesome.js"),
]


def weights_pattern(repo: str) -> str:
    """Download only ONE copy of the weights (safetensors if the repo has it)."""
    files = HfApi().list_repo_files(repo)
    return "*.safetensors" if any(f.endswith(".safetensors") for f in files) else "pytorch_model.bin"


def main():
    ap = argparse.ArgumentParser(description="Download models for offline use.")
    ap.add_argument("--whisper", choices=WHISPER_REPOS, default="large-v3")
    ap.add_argument("--translator", choices=TRANSLATOR_REPOS, default="600m")
    args = ap.parse_args()
    MODELS_DIR.mkdir(exist_ok=True)

    wdir = MODELS_DIR / f"whisper-{args.whisper}"
    print(f"Downloading speech model '{args.whisper}' -> {wdir}")
    snapshot_download(repo_id=WHISPER_REPOS[args.whisper], local_dir=wdir,
                      allow_patterns=["*.json", "*.bin", "*.txt"])

    repo, folder = TRANSLATOR_REPOS[args.translator]
    tdir = MODELS_DIR / folder
    print(f"Downloading translation model '{repo}' -> {tdir}")
    snapshot_download(repo_id=repo, local_dir=tdir,
                      allow_patterns=["*.json", "*.model", weights_pattern(repo)])

    VENDOR_DIR.mkdir(exist_ok=True)
    for url, name in PLAYER_FILES:
        print(f"Downloading {name} -> {VENDOR_DIR / name}")
        urllib.request.urlretrieve(url, VENDOR_DIR / name)

    print("\nDone. You can turn the internet OFF now. Start the app with:")
    print("  python app.py")


if __name__ == "__main__":
    main()
