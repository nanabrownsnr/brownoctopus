from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from evals.toolret.adapter import tool_embedding_text
from brown_octopus.index_store import save_embeddings, save_metadata, save_tools


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def build_toolret_index(
    tools: list[dict[str, Any]],
    output_dir: str | Path,
    dataset_hash: str,
    tool_hash: str,
    batch_size: int = 8,
    shard_size: int = 1000,
) -> dict[str, Any]:
    """Build a resumable ToolRet index in 1,000-tool shards.

    Each completed shard is durable before the next shard starts. This keeps a
    failed long CPU build resumable and makes progress visible outside tqdm.
    """
    output_dir = Path(output_dir)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "build_manifest.json"
    manifest = {
        "benchmark": "ToolRet",
        "embedding_model": MODEL_NAME,
        "tool_count": len(tools),
        "dataset_hash": dataset_hash,
        "tool_universe_hash": tool_hash,
        "representation": "ToolRet doc name + description",
        "shard_size": shard_size,
        "batch_size": batch_size,
        "completed_shards": [],
    }
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(manifest[key] == prior.get(key) for key in (
            "benchmark", "embedding_model", "tool_count", "dataset_hash", "tool_universe_hash", "representation", "shard_size", "batch_size"
        )):
            manifest.update(prior)
        else:
            raise RuntimeError("Existing ToolRet build manifest does not match this corpus/configuration")

    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    device = str(model.device)
    start = time.perf_counter()
    shard_count = (len(tools) + manifest["shard_size"] - 1) // manifest["shard_size"]
    completed = set(manifest.get("completed_shards", []))
    print(f"ToolRet resumable build: {len(tools)} tools, {shard_count} shards", flush=True)
    for shard_number in range(shard_count):
        shard_name = f"shard_{shard_number:04d}.pt"
        if shard_number in completed and (shard_dir / shard_name).exists():
            print(f"Skipping completed shard {shard_number + 1}/{shard_count}", flush=True)
            continue
        begin = shard_number * manifest["shard_size"]
        end = min(len(tools), begin + manifest["shard_size"])
        print(f"Encoding shard {shard_number + 1}/{shard_count}: tools {begin}:{end}", flush=True)
        shard_embeddings = model.encode(
            [tool_embedding_text(tool) for tool in tools[begin:end]],
            convert_to_tensor=True,
            normalize_embeddings=True,
            batch_size=manifest["batch_size"],
            show_progress_bar=False,
        ).cpu()
        temporary = shard_dir / f".{shard_name}.tmp"
        torch.save(shard_embeddings, temporary)
        temporary.replace(shard_dir / shard_name)
        completed.add(shard_number)
        manifest["completed_shards"] = sorted(completed)
        manifest["last_completed_at"] = time.time()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Completed shard {shard_number + 1}/{shard_count}", flush=True)

    if len(completed) != shard_count:
        raise RuntimeError("ToolRet build ended before all shards completed")

    embeddings = torch.cat(
        [torch.load(shard_dir / f"shard_{i:04d}.pt", weights_only=True) for i in range(shard_count)]
    )
    build_ms = (time.perf_counter() - start) * 1000.0
    save_tools(output_dir, tools)
    save_embeddings(output_dir, embeddings)
    metadata = {
        "version": 1,
        "benchmark": "ToolRet",
        "embedding_model": MODEL_NAME,
        "embedding_dimension": int(embeddings.shape[1]),
        "tool_count": len(tools),
        "dataset_hash": dataset_hash,
        "tool_universe_hash": tool_hash,
        "embedding_representation": "ToolRet doc name + description",
        "build_time_ms": build_ms,
        "shard_size": manifest["shard_size"],
        "batch_size": manifest["batch_size"],
        "shard_count": shard_count,
        "device": device,
    }
    save_metadata(output_dir, metadata)
    metadata["index_size_bytes"] = sum(
        path.stat().st_size for path in output_dir.glob("*") if path.is_file()
    )
    save_metadata(output_dir, metadata)
    return metadata


def calibration(
    tools: list[dict[str, Any]],
    sample_size: int = 100,
    batch_sizes: tuple[int, ...] = (8, 16, 32),
) -> list[dict[str, Any]]:
    """Measure short-run throughput before committing to a full build."""
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    texts = [tool_embedding_text(tool) for tool in tools[:sample_size]]
    results = []
    for batch_size in batch_sizes:
        start = time.perf_counter()
        model.encode(texts, convert_to_tensor=True, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
        elapsed = time.perf_counter() - start
        result = {"batch_size": batch_size, "sample_size": len(texts), "elapsed_seconds": elapsed, "tools_per_second": len(texts) / elapsed}
        print(json.dumps(result), flush=True)
        results.append(result)
    return results
