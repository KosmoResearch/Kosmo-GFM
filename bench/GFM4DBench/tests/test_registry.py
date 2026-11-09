from pathlib import Path

import pytest

from eval4d.registry import build_specs, find_registry, load_registry

BENCH_DIR = Path(__file__).resolve().parents[1]
PROJECT_REGISTRY = BENCH_DIR / "configs" / "subsets.yaml"
PACKAGED_REGISTRY = BENCH_DIR / "eval4d" / "configs" / "subsets.yaml"


def test_registry_copies_match():
    project = load_registry(PROJECT_REGISTRY)
    packaged = load_registry(PACKAGED_REGISTRY)
    assert project.keys() == packaged.keys()
    for name in project:
        assert project[name] == packaged[name]


def test_release_layout_is_nested():
    definitions = load_registry(PROJECT_REGISTRY)
    assert definitions.keys() == {
        "adt", "pstudio", "po", "ds", "synthverse",
    }
    for name in ("adt", "pstudio", "po", "ds"):
        definition = definitions[name]
        assert definition.data_root == "Tapvid3d_mini"
        assert definition.json.startswith("Tapvid3d_mini/Tapvid3d_mini_extracted/")
        assert definition.json.endswith("_mini.json")
        assert definition.reference_queries

    definition = definitions["synthverse"]
    assert definition.data_root == "."
    assert definition.json.startswith("SynthVerse/")
    assert definition.coord_convention == "pointodyssey"
    assert definition.reference_queries


def test_build_specs_resolves_release_layout(tmp_path):
    definitions = load_registry(PROJECT_REGISTRY)
    specs = build_specs(definitions, ["adt"], tmp_path)
    spec = specs[0]
    assert spec.json_path == (
        tmp_path
        / "Tapvid3d_mini"
        / "Tapvid3d_mini_extracted"
        / "adt_mini.json"
    )
    assert spec.resolved_data_root() == tmp_path / "Tapvid3d_mini"


def test_build_specs_resolves_synthverse(tmp_path):
    definitions = load_registry(PROJECT_REGISTRY)
    spec = build_specs(definitions, ["synthverse"], tmp_path)[0]
    assert spec.json_path == tmp_path / "SynthVerse" / "synthverse_mini.json"
    assert spec.resolved_data_root() == tmp_path
    assert spec.coord_convention == "pointodyssey"
    assert spec.dynamic_query_threshold == 0.01


def test_find_registry_prefers_project_copy(monkeypatch):
    monkeypatch.chdir(BENCH_DIR)
    assert find_registry().resolve() == PROJECT_REGISTRY.resolve()


def test_unknown_subset_is_rejected(tmp_path):
    definitions = load_registry(PROJECT_REGISTRY)
    with pytest.raises(KeyError, match="Unknown subset"):
        build_specs(definitions, ["missing"], tmp_path)
