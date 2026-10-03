"""Download only the pilot's three pinned GGUF files, without HF inference credits."""
import hashlib
import urllib.request

from utils.config import PROJECT_ROOT, load_yaml

MODEL_DIR = PROJECT_ROOT / ".local-models" / "local_240"


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    weights = load_yaml(PROJECT_ROOT / "config" / "local_models.yaml")["weights"]
    for key, entry in weights.items():
        destination = MODEL_DIR / entry["filename"]
        if destination.exists():
            if destination.stat().st_size == entry["size"] and file_hash(destination) == entry["sha256"]:
                print(f"Verified existing {key}", flush=True)
                continue
            raise RuntimeError(f"Existing file has wrong checksum: {destination}")
        url = f"https://huggingface.co/{entry['repository']}/resolve/{entry['revision']}/{entry['filename']}"
        partial = destination.with_suffix(".gguf.part")
        print(f"Downloading {key}: {entry['size'] / 1e6:.0f} MB", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
            total, next_update = 0, 100 * 1024 * 1024
            while block := response.read(8 * 1024 * 1024):
                stream.write(block)
                total += len(block)
                if total >= next_update:
                    print(f"{key}: {total / entry['size']:.0%}", flush=True)
                    next_update += 100 * 1024 * 1024
        if partial.stat().st_size != entry["size"] or file_hash(partial) != entry["sha256"]:
            raise RuntimeError(f"Downloaded checksum mismatch: {partial}")
        partial.replace(destination)
        print(f"Verified {key}", flush=True)


if __name__ == "__main__":
    main()
