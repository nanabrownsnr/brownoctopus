import json
import subprocess
import sys
from pathlib import Path


def test_indexing_benchmark_script_reports_incremental_work():
    repository = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "benchmark_indexing.py"),
            "--sizes",
            "3",
            "--json",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)[0]
    assert result["create"]["embedded_count"] == 3
    assert result["unchanged_update"]["embedded_count"] == 0
    assert result["incremental_add"]["embedded_count"] == 1
    assert result["incremental_add"]["added_count"] == 1
    assert result["changes"]["1%"]["embedded_count"] == 1
    assert result["changes"]["10%"]["embedded_count"] == 1
    assert result["changes"]["100%"]["embedded_count"] == 3
    assert result["create"]["snapshot_bytes"] > 0
    assert result["create"]["python_peak_bytes"] > 0
