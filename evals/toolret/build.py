"""Calibration and resumable index-build entry point for ToolRet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.toolret.adapter import load_toolret_tools
from evals.toolret.index import build_toolret_index, calibration, canonical_hash


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "data" / "external" / "toolret-release"
INDEX = ROOT / "data" / "indexes" / "toolret_smoke"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=1000)
    args = parser.parse_args()
    tools, sources = load_toolret_tools(RELEASE)
    tool_hash = canonical_hash(tools)
    if args.calibrate:
        print(json.dumps({"tool_count": len(tools), "tool_hash": tool_hash}, indent=2), flush=True)
        calibration(tools)
        return
    build_toolret_index(
        tools,
        INDEX,
        dataset_hash="toolret_smoke_queries",
        tool_hash=tool_hash,
        batch_size=args.batch_size,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()
