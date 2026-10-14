"""Generate dependency-free SVG plots from retrieval benchmark JSON/CSV.

The script intentionally uses only the standard library so benchmark analysis
does not add a plotting dependency to Brown Octopus or its test environment.
It accepts the JSON emitted by ``scripts/benchmark_retrieval.py`` and writes a
small HTML report containing SVG charts.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Expected a benchmark JSON list or an object with 'results'.")


def _numeric(rows: list[dict], x: str, y: str) -> list[tuple[float, float]]:
    points = []
    for row in rows:
        try:
            points.append((float(row[x]), float(row[y])))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(points)


def _svg(points: list[tuple[float, float]], title: str, x_label: str, y_label: str) -> str:
    width, height = 760, 360
    left, top, right, bottom = 70, 35, 25, 60
    plot_w, plot_h = width - left - right, height - top - bottom
    if not points:
        return f'<svg viewBox="0 0 {width} {height}"><text x="20" y="30">{html.escape(title)}: no numeric data</text></svg>'
    xmin, xmax = min(x for x, _ in points), max(x for x, _ in points)
    ymin, ymax = min(y for _, y in points), max(y for _, y in points)
    if xmin == xmax:
        xmax = xmin + 1
    if ymin == ymax:
        ymax = ymin + 1
    def px(x): return left + (x - xmin) / (xmax - xmin) * plot_w
    def py(y): return top + (ymax - y) / (ymax - ymin) * plot_h
    path = " ".join(("M" if i == 0 else "L") + f" {px(x):.1f} {py(y):.1f}" for i, (x, y) in enumerate(points))
    return f'''<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
      <rect width="100%" height="100%" fill="white"/>
      <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#555"/>
      <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#555"/>
      <path d="{path}" fill="none" stroke="#1464a0" stroke-width="2"/>
      <text x="{width / 2}" y="22" text-anchor="middle">{html.escape(title)}</text>
      <text x="{width / 2}" y="{height - 12}" text-anchor="middle">{html.escape(x_label)}</text>
      <text x="14" y="{height / 2}" transform="rotate(-90 14 {height / 2})" text-anchor="middle">{html.escape(y_label)}</text>
      <text x="{left}" y="{height - 40}">{xmin:g}</text>
      <text x="{left + plot_w}" y="{height - 40}" text-anchor="end">{xmax:g}</text>
      <text x="{left - 8}" y="{top + plot_h}" text-anchor="end">{ymin:g}</text>
      <text x="{left - 8}" y="{top + 4}" text-anchor="end">{ymax:g}</text>
    </svg>'''


def build_report(rows: list[dict]) -> str:
    normalized = []
    for row in rows:
        timing = row.get("timing_ms", {}) or {}
        normalized.append({
            **row,
            "size": row.get("size", row.get("index_size")),
            "profile": row.get("profile", row.get("query_profile")),
            "median_total_ms": row.get("median_total_ms", (timing.get("total") or {}).get("median_ms")),
            "median_retrieval_ms": row.get("median_retrieval_ms", (timing.get("retrieval") or {}).get("median_ms")),
            "median_active_tool_characters": row.get("median_active_tool_characters", row.get("active_tool_characters")),
        })
    rows = normalized
    charts = [
        _svg(_numeric(rows, "size", "median_total_ms"), "Retrieval time by index size", "index size", "median ms"),
        _svg(_numeric(rows, "size", "median_retrieval_ms"), "Candidate retrieval time by index size", "index size", "median ms"),
        _svg(_numeric(rows, "size", "median_active_tool_characters"), "Active context payload by index size", "index size", "characters"),
    ]
    rows_html = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in ("size", "profile", "candidate_k", "median_total_ms", "median_retrieval_ms", "median_active_tool_characters")) + "</tr>" for row in rows)
    return "<!doctype html><meta charset='utf-8'><title>Brown Octopus retrieval benchmark</title><style>body{font:14px sans-serif;max-width:1100px;margin:2rem auto}svg{width:48%;min-width:500px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px}</style><h1>Brown Octopus retrieval benchmark</h1><div>" + "".join(charts) + f"</div><table><thead><tr><th>size</th><th>profile</th><th>candidate_k</th><th>total ms</th><th>retrieval ms</th><th>active chars</th></tr></thead><tbody>{rows_html}</tbody></table>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("retrieval-benchmark.html"))
    args = parser.parse_args()
    rows = _load_rows(args.input)
    args.output.write_text(build_report(rows), encoding="utf-8")
    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
