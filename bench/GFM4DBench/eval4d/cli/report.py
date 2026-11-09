#!/usr/bin/env python3
"""Build a benchmark table from one or more evaluation runs.

Every ``--model NAME=PATH`` points at an evaluation output directory; the
directory must contain one ``<subset>/summary.json`` per subset.

Example::

    gfm-4d-report --task dynamic-points \\
        --model baseline=tmp/eval_baseline --model ours=tmp/eval_ours \\
        --subset adt --subset pstudio --output tmp/table.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..report.tables import ModelRow, SubsetColumn, dynamic_points_markdown, scene_flow_markdown
from .common import TASKS

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser of ``gfm-4d-report``."""
    parser = argparse.ArgumentParser(
        prog="gfm-4d-report",
        description="Build a WorldTrack benchmark table from evaluation summaries.",
    )
    parser.add_argument("--task", required=True, choices=TASKS, help="Which protocol the runs used.")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        required=True,
        help="A model entry: NAME=PATH_TO_EVAL_OUTPUT_DIR. Repeat for several models.",
    )
    parser.add_argument(
        "--subset",
        action="append",
        dest="subsets",
        default=[],
        help="Subset column, in display order. Defaults to every subset that is found.",
    )
    parser.add_argument("--output", default="benchmark_table.md", help="Output markdown file.")
    parser.add_argument(
        "--tapvid3d-subsets",
        default="adt,pstudio,po,ds",
        help="Comma-separated subsets whose average is query-weighted.",
    )
    parser.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Omit the GT-dynamic columns of the Scene Flow table.",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Subset registry used to resolve the display labels.",
    )
    return parser


def _parse_model(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), Path(path.strip())


def _load_subsets(root: Path) -> Dict[str, dict]:
    summaries: Dict[str, dict] = {}
    for path in sorted(root.glob("*/summary.json")):
        summaries[path.parent.name] = _read(path)
    direct = root / "summary.json"
    if direct.is_file():
        payload = _read(direct)
        for name, subset_summary in (payload.get("subsets") or {}).items():
            summaries.setdefault(name, subset_summary)
    return summaries


def _read(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _labels(subsets: List[str], registry: Optional[str]) -> Dict[str, str]:
    if not registry:
        return {name: name for name in subsets}
    from ..registry import find_registry, load_registry

    try:
        definitions = load_registry(find_registry(registry))
    except (FileNotFoundError, ImportError, ValueError):
        return {name: name for name in subsets}
    return {
        name: definitions[name].display_name if name in definitions else name
        for name in subsets
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point of ``gfm-4d-report``."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    rows: List[ModelRow] = []
    for spec in args.models:
        name, path = _parse_model(spec)
        subsets = _load_subsets(path)
        if not subsets:
            logging.warning("No subset summaries found in %s", path)
        rows.append(ModelRow(name=name, subsets=subsets))

    order = list(args.subsets)
    if not order:
        seen: List[str] = []
        for row in rows:
            for name in row.subsets:
                if name not in seen:
                    seen.append(name)
        order = seen
    if not order:
        logging.error("No subsets to report.")
        return 1

    labels = _labels(order, args.registry)
    columns = [SubsetColumn(key=name, label=labels.get(name, name)) for name in order]

    if args.task == "dynamic-points":
        tapvid = [s for s in args.tapvid3d_subsets.split(",") if s]
        markdown = dynamic_points_markdown(rows, columns, tapvid3d_subsets=tapvid)
    else:
        markdown = scene_flow_markdown(rows, columns, include_dynamic=not args.no_dynamic)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown + "\n", encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
