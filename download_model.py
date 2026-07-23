import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import GGUF_PATH, GGUF_REPO, GGUF_FILE, MODEL_DIR, MODEL_ID, MODEL_SIZE


def download_model():
    """Download the model GGUF. Downloads once, no re-checks."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=GGUF_REPO,
        filename=GGUF_FILE,
        local_dir=os.path.dirname(GGUF_PATH),
    )
    if os.path.abspath(path) != os.path.abspath(GGUF_PATH):
        os.rename(path, GGUF_PATH)
    print(f"Downloaded {MODEL_SIZE} model to {GGUF_PATH}")
    print(f"Size: {os.path.getsize(GGUF_PATH)/1e9:.2f} GB")


if __name__ == "__main__":
    download_model()