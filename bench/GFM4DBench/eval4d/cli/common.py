"""Shared command-line helpers."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..protocol import get_task
from ..protocol.base import BaseTask
from ..registry import SubsetDefinition, SubsetSpec, build_specs, find_registry, load_registry

__all__ = [
    "TASKS",
    "parse_dynamic_threshold",
    "add_subset_arguments",
    "resolve_specs",
    "validate_specs",
    "make_task",
    "subset_labels",
]

#: The evaluation protocols this package implements.
TASKS = ("dynamic-points", "scene-flow")

#: Default location of the extracted data package (project checkout only).
DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "4dbenchmark_data"


def _default_data_root() -> Optional[str]:
    env = os.environ.get("GFM4D_DATA_ROOT", "").strip()
    if env:
        return env
    if DEFAULT_DATA_ROOT.is_dir():
        return str(DEFAULT_DATA_ROOT)
    return None


def parse_dynamic_threshold(value: Optional[str]) -> Optional[float]:
    """Parse a dynamic-query threshold.

    ``None``, an empty string and ``none`` all mean "no threshold"; every other
    value is parsed as a float (metres).
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("none", "off", "false", "no"):
        return None
    return float(text)


def add_subset_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the subset-selection arguments to a parser."""
    parser.add_argument(
        "--subset",
        action="append",
        dest="subsets",
        default=[],
        help="Subset name from the registry. Repeat for several subsets.",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Path to the subset registry (YAML). Defaults to configs/subsets.yaml.",
    )
    parser.add_argument(
        "--data-root",
        default=_default_data_root(),
        help=(
            "Dataset root containing Tapvid3d_mini/. Defaults to GFM4D_DATA_ROOT, "
            "then bench/GFM4DBench/data/4dbenchmark_data when it exists."
        ),
    )
    parser.add_argument(
        "--json",
        action="append",
        dest="json_paths",
        default=[],
        help="Sequence JSON to evaluate directly, bypassing the registry. Repeatable.",
    )
    parser.add_argument(
        "--subset-name",
        action="append",
        dest="subset_names",
        default=[],
        help="Name of the matching --json. Defaults to the JSON file stem.",
    )
    parser.add_argument(
        "--coord-convention",
        default=None,
        choices=("opend4rt", "pointodyssey"),
        help="Override the coordinate convention of every subset.",
    )
    parser.add_argument(
        "--dynamic-query-threshold",
        default=None,
        help=(
            "Keep only the frame-0 queries whose GT motion exceeds this "
            "threshold (metres). Use 'none' to keep every frame-0 query."
        ),
    )


def make_task(name: str) -> BaseTask:
    """Instantiate a task by name."""
    return get_task(name)


def _spec_from_json(
    json_path: Path,
    name: str,
    data_root: Optional[Path],
    coord_convention: Optional[str],
    threshold: Optional[float],
) -> SubsetSpec:
    return SubsetSpec(
        name=name,
        json_path=json_path,
        data_root=data_root,
        coord_convention=coord_convention or "opend4rt",
        dynamic_query_threshold=threshold,
    )


def resolve_specs(
    args: argparse.Namespace,
) -> List[SubsetSpec]:
    """Resolve the subsets to evaluate from the command line.

    Either the registry is used (``--subset``) or the sequence JSONs are given
    directly (``--json``).
    """
    threshold = parse_dynamic_threshold(getattr(args, "dynamic_query_threshold", None))
    convention = getattr(args, "coord_convention", None)

    json_paths = [Path(p) for p in getattr(args, "json_paths", []) or []]
    if json_paths:
        names = list(getattr(args, "subset_names", []) or [])
        if names and len(names) != len(json_paths):
            raise ValueError("--subset-name count must match --json count")
        data_root = Path(args.data_root) if args.data_root else None
        specs: List[SubsetSpec] = []
        for index, json_path in enumerate(json_paths):
            name = names[index] if index < len(names) else json_path.stem
            specs.append(
                _spec_from_json(json_path, name, data_root, convention, threshold)
            )
        return specs

    subsets = list(getattr(args, "subsets", []) or [])
    if not subsets:
        raise ValueError(
            "No subset selected: pass --subset NAME (registry) or --json PATH."
        )
    if not args.data_root:
        raise ValueError(
            "--data-root is required when the subsets come from the registry. "
            "Set GFM4D_DATA_ROOT to the extracted 4dbenchmark_data directory "
            f"(default: {DEFAULT_DATA_ROOT})."
        )

    registry_path = find_registry(getattr(args, "registry", None))
    logging.info("Using subset registry: %s", registry_path)
    definitions = load_registry(registry_path)
    specs = build_specs(definitions, subsets, args.data_root)
    for spec in specs:
        if convention:
            spec.coord_convention = convention
        if getattr(args, "dynamic_query_threshold", None) is not None:
            spec.dynamic_query_threshold = threshold
    return specs


def validate_specs(specs: Sequence[SubsetSpec]) -> None:
    """Fail early with an actionable message when the data layout is wrong.

    Checks that every subset JSON exists and that the frame paths of the first
    sequence resolve under the subset's dataset root. This catches the common
    mistake of pointing ``--data-root`` one level too deep or too shallow.
    """
    from ..data.sequences import load_mf_files, resolve_worldtrack_path

    for spec in specs:
        if not spec.json_path.is_file():
            raise FileNotFoundError(
                f"Sequence JSON not found: {spec.json_path}\n"
                "Check --data-root (or GFM4D_DATA_ROOT)."
            )
        mf = load_mf_files(spec.json_path)
        if not mf:
            raise ValueError(f"No sequences in {spec.json_path}")
        sequence_name = next(iter(mf))
        frames = mf[sequence_name]
        if not frames:
            raise ValueError(
                f"The first sequence {sequence_name!r} in {spec.json_path} has no frames"
            )
        view = frames[0]["views"][0]
        try:
            resolve_worldtrack_path(
                view["rgb"],
                data_root=spec.resolved_data_root(),
                json_path=spec.json_path,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Could not resolve the frame paths of subset {spec.name!r}.\n"
                f"  JSON:         {spec.json_path}\n"
                f"  Dataset root: {spec.resolved_data_root()}\n"
                "Expected layout: Tapvid3d_mini/Tapvid3d_mini_extracted/... for "
                "the TapVid3D subsets or SynthVerse/converted/... for SynthVerse."
            ) from exc


def subset_labels(
    specs: Sequence[SubsetSpec],
    definitions: Optional[Dict[str, SubsetDefinition]] = None,
) -> Dict[str, str]:
    """A display label per subset."""
    labels: Dict[str, str] = {}
    for spec in specs:
        if definitions and spec.name in definitions:
            labels[spec.name] = definitions[spec.name].display_name
        else:
            labels[spec.name] = spec.name
    return labels
