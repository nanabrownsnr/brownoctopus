"""Smoke-check an installed Brown Octopus wheel from a consumer environment.

Run this from a clean environment after installing the built wheel. It does
not download models or build a capability index.
"""

import subprocess
import sys
from importlib.metadata import version
from shutil import which


def main() -> None:
    from brown_octopus import Octopus

    assert Octopus is not None
    assert version("brown-octopus") == "0.3.0"

    command = which("brown-octopus")
    if command:
        cli = [command]
    else:
        cli = [sys.executable, "-m", "brown_octopus.cli"]

    help_output = subprocess.run(
        [*cli, "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    setup_help = subprocess.run(
        [*cli, "setup-models", "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "Dynamic capability context management" in help_output
    assert "en_core_web_trf" in setup_help
    assert "Qwen/Qwen3-Embedding-0.6B" in setup_help
    print("Installed Brown Octopus wheel smoke validation passed.")


if __name__ == "__main__":
    main()
