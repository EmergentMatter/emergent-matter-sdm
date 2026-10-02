"""sdm_ext: the single-wedge fold fast path.

Load-bearing: it re-proves that skipping stock 0.4.0's two extra
neighbour-wedge evaluations does not change this plate's field.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from software_defined_matter import MaterialRegion, Part
from software_defined_matter.sdf.compile import make_sdf_closure

from disco_lens.optics import layout_sites, solve_site
from disco_lens.parameters import DiscoLensParameters, raise_if_invalid
from disco_lens.sdf_assembly import build_tree

# ---------------------------------------------------------------------------
# The fold fast path must agree with stock sdm-core, point for point.
# ---------------------------------------------------------------------------


def _fold_closure(tree, b_stock: bool):
    """Compile ``tree`` through either the fast path or stock sdm-core."""
    from software_defined_matter.sdf import compile as _compile

    from disco_lens import sdm_ext

    keep = _compile._compile_transform
    if b_stock:
        _compile._compile_transform = sdm_ext._STOCK_COMPILE_TRANSFORM
    try:
        part = Part(name="t")
        part.add_material(MaterialRegion(material_id=1, name="m", sdf_tree=tree))
        return make_sdf_closure(tree, part)
    finally:
        _compile._compile_transform = keep


@pytest.mark.parametrize("b_quadrant", [True, False])
def test_fold_fast_path_matches_stock(b_quadrant):
    """sdm_ext skips the two neighbour-wedge evaluations stock 0.4.0 added.

    That is only sound while every pad stays strictly inside its wedge,
    which is what the even grid's 0.3 mm seam clearance buys. Probe both
    seam planes and the wedge diagonals: the fast path must never report
    MORE room than stock, which is the direction that walks a sphere
    tracer through a surface.
    """
    p = DiscoLensParameters()
    raise_if_invalid(p)
    sites = layout_sites(p)
    for s in sites:
        solve_site(p, s, min(p.d_hub_height + 0.55 * s.d_radius, p.d_dog_top_height - 5.0))
    tree = build_tree(p, sites, b_quadrant=b_quadrant)

    rng = np.random.default_rng(7)
    d_h = p.d_plate_half
    d_lo = 0.0 if b_quadrant else -d_h
    n = 4000
    diag = rng.uniform(d_lo, d_h, n)
    pts = np.vstack(
        [
            # a band hugging each seam plane, where folding is least exact
            np.column_stack(
                [rng.uniform(-1.5, 1.5, n), rng.uniform(d_lo, d_h, n), rng.uniform(-8.0, 25.0, n)]
            ),
            np.column_stack(
                [rng.uniform(d_lo, d_h, n), rng.uniform(-1.5, 1.5, n), rng.uniform(-8.0, 25.0, n)]
            ),
            # and the 45 degree wedge diagonals
            np.column_stack([diag + rng.uniform(-1.5, 1.5, n), diag, rng.uniform(-8.0, 25.0, n)]),
        ]
    ).astype(np.float32)
    q = jnp.asarray(pts)

    d_fast = np.asarray(_fold_closure(tree, b_stock=False)(q))
    d_stock = np.asarray(_fold_closure(tree, b_stock=True)(q))

    assert not np.any(d_fast > d_stock + 1e-4), (
        "fold fast path over-reports distance: a pad now crosses a wedge "
        "boundary, so drop the fast path in sdm_ext rather than this test"
    )
    assert np.abs(d_fast - d_stock).max() < 1e-4
    assert np.array_equal(np.sign(d_fast), np.sign(d_stock))
