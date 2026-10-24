"""Fetch the four pinned assets; validate hashes before replacing local files."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "data/manifests/retail_benchmark_manifest.json").read_text())
    destination = ROOT / ".cache/tau2-retail"
    destination.mkdir(parents=True, exist_ok=True)
    for name, asset in manifest["assets"].items():
        path = destination / name
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == asset["sha256"]:
            print(f"Verified {name}")
            continue
        url = (
            f"https://raw.githubusercontent.com/sierra-research/tau2-bench/"
            f"{manifest['commit']}/{asset['upstream_path']}"
        )
        with urlopen(url, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != asset["sha256"]:
            raise ValueError(f"Source hash mismatch: {name}")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        print(f"Fetched and verified {name}")


if __name__ == "__main__":
    main()
