from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class RunSummary:
    run_name: str
    architecture: str
    bit_width: str
    ensemble_size: int
    test_sparse_acc: float
    mtime: datetime


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _load_json(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")
    return data


def _format_width(cfg: dict) -> str:
    width = cfg.get("model", {}).get("width")
    if isinstance(width, list):
        return "[" + ", ".join(str(v) for v in width) + "]"
    return str(width)


def _collect_rows(outputs_dir: Path, only_today: bool) -> list[RunSummary]:
    rows: list[RunSummary] = []
    today = datetime.now().date()

    for metrics_path in outputs_dir.glob("*/bundle/metrics.json"):
        run_dir = metrics_path.parent.parent
        if "jsc" not in run_dir.name:
            continue

        config_path = run_dir / "bundle" / "config.yaml"
        if not config_path.exists():
            continue

        mtime = datetime.fromtimestamp(run_dir.stat().st_mtime)
        if only_today and mtime.date() != today:
            continue

        try:
            cfg = _load_yaml(config_path)
            metrics = _load_json(metrics_path)
        except Exception:
            continue

        test_acc = metrics.get("test/sparse_acc")
        if not isinstance(test_acc, (int, float)):
            continue

        quant = cfg.get("quant", {})
        ensemble = cfg.get("ensemble", {})
        rows.append(
            RunSummary(
                run_name=str(cfg.get("run", {}).get("name", run_dir.name)),
                architecture=_format_width(cfg),
                bit_width=f"w{quant.get('weight_total_bits', '?')}/a{quant.get('activation_total_bits', '?')}",
                ensemble_size=int(ensemble.get("size", 1)),
                test_sparse_acc=float(test_acc),
                mtime=mtime,
            )
        )

    rows.sort(key=lambda r: r.mtime)
    return rows


def _print_table(rows: list[RunSummary]) -> None:
    if not rows:
        print("No JSC runs found for selected date filter.")
        return

    headers = ["run_name", "architecture", "bit_width", "ensemble_size", "test/sparse_acc"]
    table_rows = [
        [
            row.run_name,
            row.architecture,
            row.bit_width,
            str(row.ensemble_size),
            f"{row.test_sparse_acc:.6f}",
        ]
        for row in rows
    ]
    widths = [len(h) for h in headers]
    for r in table_rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, r, strict=False)]

    def fmt(cells: list[str]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=False))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for r in table_rows:
        print(fmt(r))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize JSC run results from outputs/")
    parser.add_argument("--outputs-dir", default="outputs", help="Root outputs directory")
    parser.add_argument(
        "--all-dates",
        action="store_true",
        help="Include runs from all dates (default: only today's runs)",
    )
    args = parser.parse_args()

    rows = _collect_rows(Path(args.outputs_dir), only_today=not args.all_dates)
    _print_table(rows)


if __name__ == "__main__":
    main()
