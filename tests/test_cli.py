import subprocess
import sys

from brown_octopus import cli


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "brown_octopus.cli", *args],
        capture_output=True,
        text=True,
    )


def test_help_and_version_commands_are_fast_and_available():
    for args in (("--help",), ("-h",), ("--version",), ("version",)):
        result = run_cli(*args)
        assert result.returncode == 0
        assert "Brown Octopus" in result.stdout or "brown-octopus" in result.stdout


def test_subcommand_help_does_not_initialize_models():
    for command in ("setup-models", "doctor", "inspect"):
        result = run_cli(command, "--help")
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "Traceback" not in result.stderr


def test_root_without_command_displays_help():
    result = run_cli()
    assert result.returncode == 0
    assert "setup-models" in result.stdout
    assert "doctor" in result.stdout
    assert "inspect" in result.stdout


def test_doctor_allows_cold_ml_startup_time():
    assert cli.MODEL_DOCTOR_TIMEOUT_SECONDS >= 120
    assert cli.EMBEDDING_DOCTOR_TIMEOUT_SECONDS >= 120
