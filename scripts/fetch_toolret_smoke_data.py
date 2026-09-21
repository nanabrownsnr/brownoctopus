"""Fetch the official ToolRet smoke shards; generated data is gitignored."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "toolret-release"
BASE_QUERIES = "https://huggingface.co/datasets/mangopy/ToolRet-Queries/resolve/main/"
BASE_TOOLS = "https://huggingface.co/datasets/mangopy/ToolRet-Tools/resolve/main/"
FILES = [
    "queries/apibank/queries-00000-of-00001.parquet",
    "queries/craft-math-algebra/queries-00000-of-00001.parquet",
    "queries/appbench/queries-00000-of-00001.parquet",
    "tools/web/tools-00000-of-00001.parquet",
    "tools/code/tools-00000-of-00001.parquet",
    "tools/customized/tools-00000-of-00001.parquet",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for relative in FILES:
        path = OUT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        base = BASE_QUERIES if relative.startswith("queries/") else BASE_TOOLS
        remote = relative.removeprefix("queries/").removeprefix("tools/")
        if not path.exists():
            print(f"Downloading {relative}", flush=True)
            with urlopen(base + remote) as response, path.open("wb") as output:
                output.write(response.read())
        hashes[relative] = sha256(path)
    manifest = {
        "repository": "https://github.com/mangopy/tool-retrieval-benchmark",
        "query_dataset": "mangopy/ToolRet-Queries main",
        "tool_dataset": "mangopy/ToolRet-Tools main",
        "files": hashes,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
