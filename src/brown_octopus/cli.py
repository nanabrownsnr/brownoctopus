"""Small, dependency-light command line interface for Brown Octopus."""

import argparse
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from brown_octopus.config import OctopusConfig
from brown_octopus.model_store import (
    EMBEDDING_MODEL_NAME,
)


DESCRIPTION = "Capability context management for AI agents."
MODEL_DOCTOR_TIMEOUT_SECONDS = 180
EMBEDDING_DOCTOR_TIMEOUT_SECONDS = 120


def _version() -> str:
    try:
        return version("brown-octopus")
    except PackageNotFoundError:
        return "0.4.8"


def _run_probe(code: str, timeout: int = 120) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (result.stdout or result.stderr).strip()
    return result.returncode == 0, output


def _index_details(index_path: Path) -> dict:
    details = {"path": str(index_path), "exists": index_path.exists()}
    if not index_path.exists():
        return details
    pointer = index_path / "current.json"
    active = index_path
    if pointer.exists():
        try:
            active = index_path / json.loads(pointer.read_text(encoding="utf-8"))["snapshot"]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            details["error"] = f"invalid current.json: {exc}"
            return details
    metadata_path = active / "metadata.json"
    details.update(
        {
            "active_path": str(active),
            "metadata": metadata_path.exists(),
            "tools": (active / "tools.json").exists(),
            "embeddings": (active / "embeddings.pt").exists(),
        }
    )
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            details.update(
                {
                    "index_version": metadata.get("version"),
                    "tool_count": metadata.get("tool_count"),
                    "embedding_model": metadata.get("embedding_model"),
                    "brown_octopus_version": metadata.get("brown_octopus_version"),
                }
            )
        except (OSError, json.JSONDecodeError) as exc:
            details["error"] = f"invalid metadata.json: {exc}"
    return details


def _doctor() -> int:
    config = OctopusConfig.from_env()
    package_ok, package_output = _run_probe("import brown_octopus")
    spacy_ok, _ = _run_probe(
        "from brown_octopus.analyzer import initialize_analyzer; "
        "initialize_analyzer(); print('en_core_web_trf')",
        timeout=MODEL_DOCTOR_TIMEOUT_SECONDS,
    )
    qwen_ok, _ = _run_probe(
        "from huggingface_hub import try_to_load_from_cache; "
        f"p=try_to_load_from_cache({EMBEDDING_MODEL_NAME!r}, 'config.json'); "
        "assert p is not None; print(p)",
        timeout=EMBEDDING_DOCTOR_TIMEOUT_SECONDS,
    )
    index = _index_details(config.index_path)
    index_ok = all(index.get(key, False) for key in ("metadata", "tools", "embeddings"))
    index_ok = index_ok and index.get("index_version") == 1
    healthy = package_ok and spacy_ok and qwen_ok and index_ok

    print("Brown Octopus Doctor\n")
    print("Package")
    mark = lambda ok: "[OK]" if ok else "[FAIL]"
    print(f"  {mark(package_ok)} brown-octopus {_version()}")
    print(f"  {mark(package_ok)} Python {sys.version.split()[0]}")
    print("\nModels")
    print(f"  {mark(spacy_ok)} en_core_web_trf")
    print(f"  {mark(qwen_ok)} Qwen/Qwen3-Embedding-0.6B")
    print("\nIndex")
    print(f"  {mark(index.get('exists', False))} index found: {config.index_path}")
    print(f"  {mark(index_ok)} compatible index version")
    if index.get("tool_count") is not None:
        print(f"  {mark(index_ok)} {index['tool_count']} capabilities")
    if not healthy:
        print("\nFix:")
        if not spacy_ok or not qwen_ok:
            print("  brown-octopus setup-models")
        if not index_ok:
            print("  Build or provide a valid index with: await octopus.update()")
        if not package_ok and package_output:
            print(f"  Import diagnostic: {package_output.splitlines()[-1]}")
        print("\nStatus\n  [FAIL] Brown Octopus is not ready.")
        return 1
    print("\nStatus\n  [OK] Brown Octopus is ready.")
    return 0


def _inspect(as_json: bool) -> int:
    config = OctopusConfig.from_env()
    index = _index_details(config.index_path)
    data = {
        "product": "Brown Octopus",
        "version": _version(),
        "capabilities": index.get("tool_count"),
        "index_version": index.get("index_version"),
        "index_path": str(config.index_path),
        "embedding_model": index.get("embedding_model") or EMBEDDING_MODEL_NAME,
        "analyzer": "deterministic-spacy",
        "selector": "V3 Min-4 + Bounded Max Gap",
        "ttl": config.ttl,
        "active_cap": config.active_cap,
    }
    if as_json:
        print(json.dumps(data, indent=2))
        return 0
    print("Brown Octopus\n")
    for key, label in (
        ("version", "Version"),
        ("capabilities", "Capabilities"),
        ("index_version", "Index version"),
        ("embedding_model", "Embedding"),
        ("analyzer", "Analyzer"),
        ("selector", "Selector"),
        ("ttl", "TTL"),
        ("active_cap", "Active cap"),
    ):
        print(f"{label + ':':16} {data[key]}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brown-octopus",
        usage="brown-octopus COMMAND",
        description=f"Brown Octopus - {DESCRIPTION}",
        epilog=(
            "Quick start:\n"
            "  pip install brown-octopus\n"
            "  brown-octopus setup-models\n"
            "  from brown_octopus import Octopus"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    subparsers = parser.add_subparsers(dest="command")
    setup = subparsers.add_parser(
        "setup-models",
        help="Prepare the models required by Octopus.",
        description=(
            "Download and validate en_core_web_trf and "
            "Qwen/Qwen3-Embedding-0.6B.\n"
            "Model setup is explicit; initialize() never downloads models."
        ),
    )
    setup.set_defaults(handler="setup-models")
    doctor = subparsers.add_parser("doctor", help="Check whether Brown Octopus is ready.")
    doctor.set_defaults(handler="doctor")
    inspect = subparsers.add_parser(
        "inspect", help="Inspect the local index and runtime configuration."
    )
    inspect.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    inspect.set_defaults(handler="inspect")
    version_parser = subparsers.add_parser("version", help="Display version information.")
    version_parser.set_defaults(handler="version")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    try:
        if handler == "setup-models":
            from brown_octopus.setup_models import main as setup_models

            setup_models()
            return 0
        if handler == "doctor":
            return _doctor()
        if handler == "inspect":
            return _inspect(args.json)
        if handler == "version":
            print(f"brown-octopus {_version()}")
            return 0
        parser.print_help()
        return 0
    except Exception as exc:
        print(f"Brown Octopus: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
