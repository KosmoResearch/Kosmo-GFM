"""The subset registry.

A *subset* is one benchmark split: a sequence JSON plus the conventions needed
to load it (coordinate convention, dynamic-query threshold). The registry is a
YAML file so that a benchmark can be described without touching the code.

Paths in the registry are **relative**. The dataset root is supplied by the
user at run time (``--data-root``); the package never ships an absolute path.

The released data package extracts to::

    <GFM4D_DATA_ROOT>/
    ├── Tapvid3d_mini/
    │   └── Tapvid3d_mini_extracted/
    │       └── adt_mini.json
    └── SynthVerse/
        └── synthverse_mini.json

Example registry entry::

    subsets:
      adt:
        label: ADT
        data_root: Tapvid3d_mini
        json: Tapvid3d_mini/Tapvid3d_mini_extracted/adt_mini.json
        coord_convention: opend4rt
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "SubsetDefinition",
    "SubsetSpec",
    "load_registry",
    "build_specs",
    "find_registry",
]


@dataclass
class SubsetDefinition:
    """One entry of the subset registry, before the dataset root is applied."""

    name: str
    json: str
    label: str = ""
    #: Path of this subset's dataset root, relative to the global data root.
    data_root: str = "."
    coord_convention: str = "opend4rt"
    #: Keep only the frame-0 queries whose GT motion exceeds this threshold.
    dynamic_query_threshold: Optional[float] = None
    #: Expected number of scored queries; used as an acceptance check.
    reference_queries: Optional[int] = None
    #: Expected number of sequences in the subset.
    reference_sequences: Optional[int] = None

    @property
    def display_name(self) -> str:
        return self.label or self.name


@dataclass
class SubsetSpec:
    """One resolved benchmark subset: a sequence JSON plus its conventions."""

    name: str
    json_path: Path
    data_root: Optional[Path] = None
    coord_convention: str = "opend4rt"
    #: Keep only the frame-0 queries whose GT motion exceeds this threshold
    #: (metres). ``None`` keeps every frame-0 query.
    dynamic_query_threshold: Optional[float] = None
    reference_queries: Optional[int] = None
    reference_sequences: Optional[int] = None

    def resolved_data_root(self) -> Path:
        """The dataset root: the explicit one, else the JSON's parent's parent."""
        if self.data_root is not None:
            return Path(self.data_root)
        return Path(self.json_path).parent.parent


def _require_yaml():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading the subset registry requires PyYAML. Install it with "
            "'pip install pyyaml'."
        ) from exc
    return yaml


def load_registry(path: Any) -> Dict[str, SubsetDefinition]:
    """Read a subset registry file.

    Args:
        path: The YAML file.

    Returns:
        The subset definitions, keyed by subset name.
    """
    yaml = _require_yaml()
    registry_path = Path(path)
    if not registry_path.is_file():
        raise FileNotFoundError(f"Subset registry not found: {registry_path}")
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    raw = payload.get("subsets", payload)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Invalid subset registry: {registry_path}")

    definitions: Dict[str, SubsetDefinition] = {}
    for name, entry in raw.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"Invalid subset entry {name!r} in {registry_path}")
        json_ref = entry.get("json")
        if not json_ref:
            raise ValueError(f"Subset {name!r} has no 'json' entry in {registry_path}")
        threshold = entry.get("dynamic_query_threshold", None)
        definitions[str(name)] = SubsetDefinition(
            name=str(name),
            json=str(json_ref),
            label=str(entry.get("label", "") or ""),
            data_root=str(entry.get("data_root", ".") or "."),
            coord_convention=str(entry.get("coord_convention", "opend4rt") or "opend4rt"),
            dynamic_query_threshold=None if threshold is None else float(threshold),
            reference_queries=(
                None
                if entry.get("reference_queries") is None
                else int(entry["reference_queries"])
            ),
            reference_sequences=(
                None
                if entry.get("reference_sequences") is None
                else int(entry["reference_sequences"])
            ),
        )
    return definitions


def build_specs(
    definitions: Mapping[str, SubsetDefinition],
    names: Sequence[str],
    data_root: Any,
    *,
    override_json: Optional[Mapping[str, Any]] = None,
) -> List[SubsetSpec]:
    """Turn registry definitions into runnable specs.

    Args:
        definitions: The registry entries.
        names: The subsets to build, in order.
        data_root: The dataset root supplied by the user.
        override_json: Per-subset JSON overrides, keyed by subset name.

    Command-line overrides of the coordinate convention and of the dynamic
    query threshold are applied by the caller on the returned specs.
    """
    root = Path(data_root)
    specs: List[SubsetSpec] = []
    for name in names:
        if name not in definitions:
            raise KeyError(f"Unknown subset {name!r}; available: {sorted(definitions)}")
        definition = definitions[name]

        json_ref = str((override_json or {}).get(name, definition.json))
        json_path = Path(json_ref)
        if not json_path.is_absolute():
            json_path = root / json_ref

        subset_root = root / definition.data_root if definition.data_root != "." else root

        specs.append(
            SubsetSpec(
                name=definition.name,
                json_path=json_path,
                data_root=subset_root,
                coord_convention=definition.coord_convention,
                dynamic_query_threshold=definition.dynamic_query_threshold,
                reference_queries=definition.reference_queries,
                reference_sequences=definition.reference_sequences,
            )
        )
    return specs


def find_registry(explicit: Optional[Any] = None) -> Path:
    """Locate the subset registry.

    Order: the explicit path, the ``EVAL4D_REGISTRY`` environment variable,
    then ``configs/subsets.yaml`` relative to the current directory, the
    project root, or the installed package.
    """
    import os

    if explicit:
        return Path(explicit)
    env = os.environ.get("EVAL4D_REGISTRY", "")
    if env:
        return Path(env)
    package_dir = Path(__file__).resolve().parent
    candidates = [
        Path("configs/subsets.yaml"),
        package_dir.parent / "configs" / "subsets.yaml",
        package_dir / "configs" / "subsets.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No subset registry found. Pass --registry, set EVAL4D_REGISTRY, or "
        "create configs/subsets.yaml."
    )
