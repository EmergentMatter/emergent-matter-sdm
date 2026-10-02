"""The primitives-showcase catalog: shape, uniqueness, membership, and
cross-file legend coverage. Pins that these hold without ``bpy`` or
``software_defined_matter`` installed -- neither is a dependency of this
package (both are imported lazily, at point of use, per ``build.py`` and
``layout.py``'s module docstrings), and this is what keeps the ``test`` CI
job non-vacuous: a regression here would otherwise only surface inside
Blender, on a machine with an sdm-core checkout.

Membership is pinned as an explicit golden set per family (see
``EXPECTED_CATALOG_NAMES`` / ``EXPECTED_CONSTRUCTION_NAMES``), not just a
count: the board's entire purpose is to show every primitive, and a
primitive silently dropped from the catalog is invisible in the rendered
PNG. A set comparison catches both a deletion and an unannounced addition,
and names the specific primitive that changed.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from sdm_showcase import build, layout

# A stand-in for _backend()'s SimpleNamespace: enough of sdm-core's geometry
# SDK, as opaque markers, to walk every CATALOG/CONSTRUCTIONS builder without
# needing the real geometry backend installed. Builders combine these with
# plain Python (list/dict literals, loops), so exercising them with markers
# both proves they run without raising and pins each entry's parameter-sweep
# arity, without asserting anything about actual geometry.
FAKE_SDK = SimpleNamespace(
    P=lambda *a, **k: ("P", a, k),
    T=lambda *a, **k: ("T", a, k),
    MOD=lambda *a, **k: ("MOD", a, k),
    LIFT=lambda *a, **k: ("LIFT", a, k),
    op=lambda *a, **k: ("op", a, k),
    loft=lambda *a, **k: ("loft", a, k),
    sweep=lambda *a, **k: ("sweep", a, k),
)


def test_build_module_imports_without_bpy_or_sdm_core() -> None:
    """`import sdm_showcase.build` must not touch the geometry backend --
    only calling _backend() (or a CATALOG builder) may."""
    assert callable(build.build_catalog)
    assert callable(build.build_constructions)


def test_layout_module_imports_without_bpy_or_sdm_core() -> None:
    """`import sdm_showcase.layout` must not touch bpy -- only calling
    build_layout() may."""
    assert callable(layout.build_layout)


def test_backend_missing_sdm_core_raises_with_an_install_hint() -> None:
    """_backend() turns the bare ImportError into a RuntimeError that tells
    the user how to get the dependency, per STYLE.md's lazy-import rule."""
    with pytest.raises(RuntimeError, match="emergent-matter-sdm-core"):
        build._backend()


# The golden set: every primitive name build.py's CATALOG must have, by
# family. A deliberate literal, not derived from build.py, so a name
# silently dropped (or added without updating this file) fails loudly
# instead of the test trivially agreeing with whatever the catalog says.
EXPECTED_CATALOG_NAMES = {
    "exact_3d": {
        "sphere", "box", "round_box", "box_frame", "torus", "capped_torus",
        "link", "cone", "plane", "hex_prism", "tri_prism", "capsule",
        "capped_cylinder", "rounded_cylinder", "capped_cone", "solid_angle",
        "cut_sphere", "ellipsoid", "octahedron", "pyramid", "helix", "screw_thread",
        "nut_bolt_washer", "fastener_optionality",
    },
    "tpms": {"gyroid", "schwarz_p", "schwarz_d", "neovius", "lidinoid"},
    "compliant": {"notch_hinge", "leaf_spring", "bellows", "serpentine", "annular_sector"},
    "profile_2d": {
        "circle_2d", "box_2d", "rounded_box_2d", "segment_2d", "trapezoid_2d",
        "uneven_capsule_2d", "polygon_2d", "bezier_2d", "bspline_2d",
    },
}  # fmt: skip

EXPECTED_CONSTRUCTION_NAMES = {
    "loft_morphing_column", "loft_twisted_vase", "loft_bezier",
    "sweep_ring", "sweep_coil", "sweep_bspline_path",
}  # fmt: skip


def _all_catalog_names() -> list[str]:
    cat = build.build_catalog(FAKE_SDK)
    names = [name for _color, items in cat.values() for name, _builder in items]
    names += [name for name, _half, _builder in build.build_constructions(FAKE_SDK)]
    return names


def test_catalog_names_are_unique_across_families_and_constructions() -> None:
    """Every STL basename build.py writes must be unique -- a collision would
    silently overwrite one primitive's mesh with another's."""
    names = _all_catalog_names()
    assert len(names) == len(set(names))


def test_catalog_family_membership_matches_the_golden_set() -> None:
    """Every family's primitive names must exactly match the golden set --
    a name added or dropped without updating both sides fails here, naming
    the specific primitive that changed, rather than surfacing only as a
    silently incomplete rendered board."""
    cat = build.build_catalog(FAKE_SDK)
    assert set(cat) == set(EXPECTED_CATALOG_NAMES)
    for family, expected_names in EXPECTED_CATALOG_NAMES.items():
        _color, items = cat[family]
        actual_names = {name for name, _builder in items}
        assert actual_names == expected_names, family


def test_construction_membership_matches_the_golden_set() -> None:
    """The Lofts & Sweeps row's names must exactly match the golden set,
    same reasoning as the catalog family check above."""
    actual_names = {name for name, _half, _builder in build.build_constructions(FAKE_SDK)}
    assert actual_names == EXPECTED_CONSTRUCTION_NAMES


def test_catalog_family_colors_are_valid_rgb_triples() -> None:
    """Each family's Base Color is a 3-tuple in [0, 1], as Blender's Principled
    BSDF expects (see layout.py's material_for)."""
    for family, (color, _items) in build.build_catalog(FAKE_SDK).items():
        assert len(color) == 3, family
        assert all(0.0 <= c <= 1.0 for c in color), family


def test_every_catalog_family_has_a_legend_label() -> None:
    """layout.py's left-margin color legend (CATEGORY_LABELS) must cover
    every family build.py's CATALOG defines -- a family added to one without
    the other would render with no legend entry, silently."""
    families = set(build.build_catalog(FAKE_SDK))
    assert families <= set(layout.CATEGORY_LABELS)


@pytest.mark.parametrize("family", list(build.build_catalog(FAKE_SDK)))
def test_catalog_builders_return_three_perturbations(family: str) -> None:
    """Every grid entry shows exactly three parameter perturbations, per this
    package's README -- a builder returning any other count would silently
    misalign build.py's PX spacing and the STL grid layout."""
    _color, items = build.build_catalog(FAKE_SDK)[family]
    for name, builder in items:
        assert len(builder()) == 3, f"{family}/{name}"


def test_construction_builders_run_without_raising() -> None:
    """The Lofts & Sweeps row's builders must run against a stub backend --
    they take no perturbation-count contract (each is one part, not three)."""
    for name, half, builder in build.build_constructions(FAKE_SDK):
        assert len(half) == 3, name
        builder()  # must not raise


def _row_z(cell: dict[str, Any]) -> float:
    return layout.row_z(cell, build_z=-0.5, prof_top_z=-0.6, prof_rows=[0, 1])


def test_row_z_stacks_profile_rows_below_the_profile_top() -> None:
    """2-D profile rows stack downward from prof_top_z; row 0 sits exactly at
    the top of the profile block."""
    assert _row_z({"category": "profile_2d", "row": 0}) == -0.6
    assert _row_z({"category": "profile_2d", "row": 1}) == pytest.approx(-0.6 - layout.PZG)


def test_row_z_places_non_profile_rows_at_a_plain_multiple_of_pzg() -> None:
    """Every other family stacks from z=0 by row index alone, independent of
    the profile block's position."""
    assert _row_z({"category": "exact_3d", "row": 0}) == 0.0
    assert _row_z({"category": "tpms", "row": 3}) == pytest.approx(-3 * layout.PZG)


def test_sc_returns_a_unit_vector() -> None:
    """sc(deg) is [sin, cos] of the angle -- used as a direction, so it must
    stay unit length."""
    x, y = build.sc(37.0)
    assert x**2 + y**2 == pytest.approx(1.0)


def test_octahedron_uses_the_primitive_not_a_csg_workaround() -> None:
    """The octahedron entry calls sdm-core's ``octahedron`` primitive
    directly. It used to be an eight-plane intersection standing in for a
    single-signed primitive; that fix has landed upstream and the
    workaround is gone, so a reintroduced ``op("intersect", ...)`` here is a
    regression, not a safety net."""
    _color, items = build.build_catalog(FAKE_SDK)["exact_3d"]
    parts = dict(items)["octahedron"]()
    assert len(parts) == 3
    for kind, args, _kwargs in parts:
        assert (kind, args[0]) == ("P", "octahedron")


def test_tpms_entries_use_period_and_min_thickness() -> None:
    """Every TPMS entry passes sdm-core's current keyword pair, in mm. The
    older ``scale``/``thickness`` pair fails at build time with a TypeError
    that only shows up once the geometry backend is installed, so pin the
    names here where the fake SDK can see them."""
    _color, items = build.build_catalog(FAKE_SDK)["tpms"]
    for _name, builder in items:
        for _kind, _args, kwargs in builder():
            assert {"period", "min_thickness", "n_periods"} <= set(kwargs)
            assert not {"scale", "thickness"} & set(kwargs)


@pytest.mark.parametrize(
    ("n", "ro", "ri"), [(5, 9, 4), (3, 7, 2)], ids=["pentagram", "triangle-star"]
)
def test_star_alternates_outer_and_inner_radius(n: int, ro: float, ri: float) -> None:
    """star() must place 2n points, alternating outer/inner radius -- an off-
    by-one here silently draws a regular polygon instead of a star."""
    pts = build.star(n, ro, ri)
    assert len(pts) == 2 * n
    r0 = (pts[0][0] ** 2 + pts[0][1] ** 2) ** 0.5
    r1 = (pts[1][0] ** 2 + pts[1][1] ** 2) ** 0.5
    assert r0 == pytest.approx(ro)
    assert r1 == pytest.approx(ri)


def test_ngon_produces_n_equally_spaced_points_on_a_circle() -> None:
    pts = build.ngon(6, 7.0)
    assert len(pts) == 6
    for x, y in pts:
        assert (x**2 + y**2) ** 0.5 == pytest.approx(7.0)


def test_helix_path_covers_the_requested_height_and_turns() -> None:
    """The path's z runs from -h/2 to h/2, and it completes the requested
    number of turns in the xy-plane."""
    path = build.helix_path(r=5.0, turns=2.0, h=10.0, n=41)
    assert path[0][2] == pytest.approx(-5.0)
    assert path[-1][2] == pytest.approx(5.0)
    assert path[0][0] == pytest.approx(path[-1][0])  # 2 full turns: same xy as start
    assert path[0][1] == pytest.approx(path[-1][1])


def test_circle_path_xz_is_closed_and_planar_in_y() -> None:
    path = build.circle_path_xz(radius=8.0, n=40)
    assert all(y == 0.0 for _x, y, _z in path)
    x0, _y0, z0 = path[0]
    assert (x0**2 + z0**2) ** 0.5 == pytest.approx(8.0)


def test_bez_ellipse_control_points_span_the_requested_axes() -> None:
    cp = build.bez_ellipse(rx=8.0, ry=5.0)
    xs = [p[0] for p in cp]
    ys = [p[1] for p in cp]
    assert max(xs) == pytest.approx(8.0)
    assert min(xs) == pytest.approx(-8.0)
    assert max(ys) == pytest.approx(5.0)
    assert min(ys) == pytest.approx(-5.0)


def test_radial_cp_samples_the_given_radius_function() -> None:
    pts = build.radial_cp(build.circ(6.0), n=12)
    assert len(pts) == 12
    for x, y in pts:
        assert (x**2 + y**2) ** 0.5 == pytest.approx(6.0)


def test_lobe_radius_oscillates_around_the_base_radius() -> None:
    """lobe(R, k, amp)'s radius function must return exactly R at a peak
    (cos = 1) and R*(1-amp) at a trough (cos = -1)."""
    r = build.lobe(R=10.0, k=1.0, amp=0.3)
    assert r(0.0) == pytest.approx(10.0 * 1.3)
    assert r(math.pi) == pytest.approx(10.0 * 0.7)
