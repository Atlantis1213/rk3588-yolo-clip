"""Download the official same-architecture reference into a separate experimental cache."""
import hashlib
from pathlib import Path
import urllib.request

SHA = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
URL = f"https://openaipublic.azureedge.net/clip/models/{SHA}/ViT-L-14.pt"
output = Path(__file__).resolve().parent / "assets/ViT-L-14.pt"
output.parent.mkdir(parents=True, exist_ok=True)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): value.update(block)
    return value.hexdigest()


if output.exists() and digest(output) == SHA:
    print("Official checkpoint already verified", flush=True)
else:
    partial = output.with_suffix(".download")
    with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as handle:
        size = int(response.headers.get("Content-Length", 0))
        downloaded, next_log = 0, 100 * 1024 * 1024
        while True:
            block = response.read(1024 * 1024)
            if not block: break
            handle.write(block); downloaded += len(block)
            if downloaded >= next_log:
                print(f"Downloaded {downloaded / 1024**2:.0f} MiB / {size / 1024**2:.0f} MiB", flush=True)
                next_log += 100 * 1024 * 1024
    if digest(partial) != SHA: raise RuntimeError("official model SHA-256 mismatch")
    partial.replace(output)
    print("Official ViT-L/14 checkpoint downloaded and SHA-256 verified", flush=True)
print("This is a reconstructed official reference; deployed historical weight provenance is unconfirmed.")
