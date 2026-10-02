"""cem.toml against the Python, in both directions.

Adapted from the CEM template, which says to copy this file verbatim
and is right about why: nothing else reads the manifest, so without
these checks it drifts into describing a design nobody is building, and
nothing fails when it does.

The manifest is what a tool reads. The Python is what runs.
"""

from __future__ import annotations

import dataclasses
import inspect
import tomllib
from pathlib import Path

import pytest

from disco_lens.parameters import DiscoLensParameters
from disco_lens.sdf_assembly import PARAM_FIELDS

ROOT = Path(__file__).resolve().parent.parent
CEM = tomllib.loads((ROOT / "cem.toml").read_text())["cem"]

DECLARED = {f.name: f for f in dataclasses.fields(DiscoLensParameters)}
DERIVED = {
    n
    for n in dir(DiscoLensParameters)
    if isinstance(getattr(DiscoLensParameters, n, None), property)
}

#: Prefix to the Python type it promises. d_ means float, not
#: millimetre: angles and ratios are d_ too, which is why every param
#: carries an explicit unit instead of inferring one from the prefix.
PREFIX_TYPE = {"d_": "float", "n_": "int", "b_": "bool", "s_": "str"}


def test_entry_point_is_not_resolved_to_read_the_manifest():
    """read_cem parses this file without importing anything. Keep it
    parseable by a plain TOML reader with no code execution."""
    assert CEM["entry_point"].count(":") == 1
    s_module, s_func = CEM["entry_point"].split(":")
    assert s_module.startswith("disco_lens.")
    assert s_func.startswith("build_")


@pytest.mark.parametrize("s_name", sorted(CEM["params"]))
def test_every_declared_param_exists_on_the_dataclass(s_name):
    assert s_name in DECLARED, (
        f"cem.toml declares {s_name!r}, which is not a field on DiscoLensParameters"
    )


@pytest.mark.parametrize("s_name", sorted(DECLARED))
def test_every_dataclass_field_is_declared(s_name):
    assert s_name in CEM["params"], (
        f"{s_name!r} is a declared field that cem.toml does not describe. "
        "A tool reading the manifest cannot see it at all"
    )


@pytest.mark.parametrize("s_name", sorted(CEM["params"]))
def test_declared_default_matches_the_dataclass(s_name):
    d_manifest = CEM["params"][s_name]["default"]
    d_python = getattr(DiscoLensParameters(), s_name)
    assert d_manifest == pytest.approx(d_python), (
        f"{s_name}: cem.toml says {d_manifest}, Python says {d_python}"
    )


@pytest.mark.parametrize("s_name", sorted(CEM["params"]))
def test_declared_type_matches_the_prefix(s_name):
    s_prefix = s_name[:2]
    assert s_prefix in PREFIX_TYPE, f"{s_name} has no type prefix"
    assert CEM["params"][s_name]["type"] == PREFIX_TYPE[s_prefix]


@pytest.mark.parametrize("s_name", sorted(CEM["params"]))
def test_every_param_carries_a_unit_a_role_and_a_doc(s_name):
    entry = CEM["params"][s_name]
    for s_key in ("unit", "role", "doc"):
        assert entry.get(s_key), f"{s_name} is missing {s_key}"
    assert entry["role"] in {"live", "re-emit", "topology", "derived", "tool"}


@pytest.mark.parametrize("s_name", sorted(CEM["params"]))
def test_bounds_bracket_the_default(s_name):
    entry = CEM["params"][s_name]
    if "bounds" not in entry:
        return
    d_lo, d_hi = entry["bounds"]
    assert d_lo <= entry["default"] <= d_hi, (
        f"{s_name}: default {entry['default']} is outside its own bounds"
    )


def test_derived_values_are_not_declared_as_params():
    """Shipping a derived value severs it from its inputs."""
    leaked = sorted(DERIVED & set(CEM["params"]))
    assert not leaked, f"derived values declared in cem.toml: {leaked}"

    shipped = sorted(DERIVED & {f for f, _, _ in PARAM_FIELDS})
    assert not shipped, f"derived value shipped in the .sdm: {shipped}"


def test_emitted_names_match_the_manifest():
    """`emits` is what lets a consumer join this CEM to its own output
    without guessing."""
    manifest = {n: e["emits"] for n, e in CEM["params"].items() if "emits" in e}
    assert manifest == {f: e for f, e, _ in PARAM_FIELDS}, (
        "cem.toml's `emits` entries and PARAM_FIELDS disagree about what the .sdm ships"
    )


@pytest.mark.parametrize("entry", PARAM_FIELDS, ids=lambda e: e[1])
def test_shipped_bounds_match_the_manifest(entry):
    """The bounds ride in the document as well as the manifest, because
    a param shipped without them renders as a control nothing can move.
    Two copies, so a test has to hold them together."""
    s_field, _, bounds = entry
    assert tuple(CEM["params"][s_field]["bounds"]) == bounds


def test_every_declared_constraint_is_actually_enforced():
    """Each constraint's name must appear in a validate() message, which
    is why those messages are prefixed with the name. A rule nothing
    checks is not a rule."""
    src = inspect.getsource(DiscoLensParameters.validate)
    for c in CEM["constraints"]:
        assert c["name"] in src, (
            f"cem.toml declares constraint {c['name']!r} that validate() never enforces"
        )


def test_every_enforced_constraint_is_declared():
    """The other direction: a rule that fires but is not declared is
    invisible to anything reading the manifest."""
    src = inspect.getsource(DiscoLensParameters.validate)
    declared = {c["name"] for c in CEM["constraints"]}
    enforced = {
        line.split('"')[1].split(":")[0] for line in src.splitlines() if 'out.append("' in line
    }
    assert enforced <= declared, (
        f"validate() enforces undeclared constraints: {sorted(enforced - declared)}"
    )


@pytest.mark.parametrize("c", CEM["constraints"], ids=lambda c: c["name"])
def test_every_constraint_carries_its_reason(c):
    """A constraint whose reason lives only in someone's head gets
    relaxed by the next person who finds it inconvenient."""
    assert c.get("why"), f"{c['name']} has no why"
    assert len(c["why"]) > 30, f"{c['name']}'s why is too thin to be useful"


def test_manifest_declares_the_sdm_core_floor_the_package_requires():
    """One place to look for which sdm-core this CEM needs."""
    s_pyproject = (ROOT / "pyproject.toml").read_text()
    assert CEM["sdm_core"].lstrip(">=") in s_pyproject
