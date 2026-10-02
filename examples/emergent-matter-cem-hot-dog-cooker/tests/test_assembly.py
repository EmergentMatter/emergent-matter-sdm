"""Compiled-SDF checks for the quadrant plate and clamp ring.

Geometry-only: sites get cheap feasible aims (no optimizer run) - these
tests probe the built solid, not the optics. The optics themselves are
covered by test_optics.py.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from software_defined_matter import MaterialRegion, Part
from software_defined_matter.sdf.compile import make_sdf_closure

from disco_lens.optics import layout_sites, solve_site
from disco_lens.parameters import DiscoLensParameters, raise_if_invalid
from disco_lens.sdf_assembly import build_ring_tree, build_tree


def _closure(tree):
    part = Part(name="t")
    part.add_material(MaterialRegion(material_id=1, name="m", sdf_tree=tree))
    return make_sdf_closure(tree, part)


@pytest.fixture(scope="module")
def design():
    p = DiscoLensParameters()
    raise_if_invalid(p)
    sites = layout_sites(p)
    for s in sites:  # feasible monotone aims; geometry only
        solve_site(p, s, min(p.d_hub_height + 0.55 * s.d_radius, p.d_dog_top_height - 5.0))
    return p, sites


@pytest.fixture(scope="module")
def quadrant(design):
    p, sites = design
    return p, _closure(build_tree(p, sites))


def sdf_at(f, pts):
    return np.asarray(f(jnp.asarray(pts, dtype=jnp.float32)))


def test_quadrant_is_one_quadrant(quadrant):
    """Nothing survives past the seam planes; Q1 slab and quarter-post do."""
    p, f = quadrant
    d = sdf_at(
        f,
        [
            [3.0, 3.0, 30.0],  # quarter-post wall
            [100.0, 100.0, -1.5],  # slab interior
            [-3.0, 3.0, 30.0],  # mirrored post: cut away
            [-100.0, 100.0, -1.5],  # Q2 slab: cut away
            [100.0, -100.0, -1.5],  # Q4 slab: cut away
            [-1.0, 100.0, 5.0],  # just across the x=0 seam
            [100.0, -1.0, 5.0],  # just across the y=0 seam
        ],
    )
    assert (d[:2] < 0.0).all()
    assert (d[2:] > 0.0).all()


def test_quarter_post_full_height_and_bore(quadrant):
    p, f = quadrant
    d_rw = (p.d_hub_diameter / 2.0 + p.d_stick_hole_diameter / 2.0) / 2.0
    d_w = d_rw / math.sqrt(2.0)
    d = sdf_at(
        f,
        [
            [d_w, d_w, p.d_hub_height - 1.0],  # post wall near the top
            [d_w, d_w, p.d_hub_height + 1.0],  # above the post: air
            [1.0, 1.0, p.d_hub_height - 1.0],  # skewer quarter-groove: air
        ],
    )
    assert d[0] < 0.0 and d[1] > 0.0 and d[2] > 0.0


def test_one_tab_with_boss_and_pinhole(quadrant):
    p, f = quadrant
    d_cx, d_cy = p.d_sight_center_x, p.d_tab_center_y
    d = sdf_at(
        f,
        [
            [d_cx + 3.0, d_cy, 10.0],  # boss wall
            [d_cx, d_cy, 10.0],  # pinhole: air
            [d_cx, d_cy, p.d_sight_boss_height - 1.0],  # boss near top (wall
            # is only Ø1.4 open at center)
            [d_cy, d_cx, 10.0],  # rotated position: NO tab
            # on the +Y edge of Q1
        ],
    )
    assert d[0] < 0.0
    assert d[1] > 0.0
    assert d[3] > 0.0


def test_nothing_past_rounded_outer_corner(quadrant):
    p, f = quadrant
    d_c = p.d_plate_half - p.d_plate_corner_radius + p.d_plate_corner_radius / math.sqrt(2.0) + 0.6
    d = sdf_at(f, [[d_c, d_c, d_z] for d_z in (-1.5, 2.0, 8.0)])
    assert (d > 0.0).all()


def test_full_tree_matches_quadrant_in_q1(design):
    """The printed quadrant IS the full assembly restricted to Q1."""
    p, sites = design
    f_quad = _closure(build_tree(p, sites, b_quadrant=True))
    f_full = _closure(build_tree(p, sites, b_quadrant=False))
    rng = np.random.default_rng(3)
    pts = np.column_stack(
        [
            rng.uniform(2.0, p.d_plate_half - 2.0, 200),
            rng.uniform(2.0, p.d_plate_half - 2.0, 200),
            rng.uniform(-p.d_base_thickness, 12.0, 200),
        ]
    )
    d_q = sdf_at(f_quad, pts)
    d_f = sdf_at(f_full, pts)
    # Interior points classify identically (distances differ near seams).
    assert ((d_q < 0.0) == (d_f < 0.0)).mean() > 0.995


def test_ring_fits_post_and_hides_in_dog_shadow(design):
    p, _ = design
    f = _closure(build_ring_tree(p))
    d_mid = (p.d_ring_bore_radius + p.d_ring_outer_radius) / 2.0
    d = sdf_at(
        f,
        [
            [d_mid, 0.0, p.d_ring_height / 2.0],  # wall
            [0.0, 0.0, p.d_ring_height / 2.0],  # bore: air
            [p.d_hub_diameter / 2.0 + 0.1, 0.0, p.d_ring_height / 2.0],
        ],
    )
    assert d[0] < 0.0
    assert d[1] > 0.0
    assert d[2] > 0.0  # post surface + slip fit stays clear
    assert p.d_ring_bore_radius > p.d_hub_diameter / 2.0
    assert 2.0 * p.d_ring_outer_radius < p.d_dog_diameter
