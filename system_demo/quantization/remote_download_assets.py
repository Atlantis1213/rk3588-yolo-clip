"""Download only fixed, verified public assets into the isolated training directory."""
import hashlib
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent / "assets"
root.mkdir(exist_ok=True)
assets = [("ViT-L-14.pt", "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836",
           "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"),
          ("dataset-resized.zip", "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2",
           "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip")]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


for filename, expected, url in assets:
    destination = root / filename
    if destination.exists():
        if digest(destination) != expected:
            raise RuntimeError("existing asset hash differs: " + filename)
        print("VERIFIED existing " + filename, flush=True)
        continue
    partial = root / (filename + ".download")
    print("DOWNLOAD " + filename, flush=True)
    candidates = [url]
    if filename == 'dataset-resized.zip':
        candidates += [
            'https://github.com/garythung/trashnet/raw/refs/heads/master/data/dataset-resized.zip',
            'https://ghfast.top/'+url]
    verified = False
    for source in candidates:
        print('DOWNLOAD source '+source,flush=True)
        result = subprocess.run(["curl.exe", "--silent", "--show-error", "--fail", "--location",
                    "--http1.1", "--retry", "2", "--retry-all-errors", "--retry-delay", "2",
                    "--connect-timeout", "15", "--max-time", "240", "--output", str(partial), source])
        if result.returncode == 0 and digest(partial) == expected:
            verified = True
            break
        print('DOWNLOAD failed or hash mismatch, trying next source',flush=True)
    if not verified:
        raise RuntimeError('no source produced the verified asset: '+filename)
    if digest(partial) != expected:
        raise RuntimeError("download SHA mismatch: " + filename)
    partial.replace(destination)
    print("VERIFIED " + filename, flush=True)
