"""disco-lens's private sdm-core EXTENSION, registered at import time.

Rob's rule: nothing of ours lives in the sdm-core checkout - it stays
pristine at origin/main. Anything this project needs beyond stock
sdm-core is defined HERE and injected into the library's registries when
``disco_lens`` is imported.

There is exactly one such thing left: a single-wedge fast path for
``canonical_sector_fold``, and it is a PERFORMANCE specialization, not a
correctness fix. See the comment on the block itself.

Upstream status (sdm-core 0.4.0, checked 2026-09-16): the jit-safety
half of the old fold patch is UPSTREAM and this module no longer carries
it. Stock ``canonical_sector_fold`` floors the sector count with
``jnp.floor`` instead of calling ``int()`` on a traced value (issue
#107), so a folded part meshes, exports and previews without help from
us. 0.4.0 also taught the GLSL emitter ``canonical_sector_fold`` and
``mirror``, so these plates preview out of process.
"""

from __future__ import annotations

from software_defined_matter.sdf import compile as _compile
from software_defined_matter.sdf import transforms as _transforms

# The stock ``_compile_transform``, captured before we wrap it. Tests use
# it to re-prove that the fold fast path below agrees with upstream.
_STOCK_COMPILE_TRANSFORM = None


def _op_kwargs_have(node, *names) -> bool:
    """True if the node authored any of ``names``, which the fast path
    does not implement (0.4.0 gave the fold ``centered`` and
    ``phase_frac``; this plate authors neither)."""
    kw = _compile._op_kwargs(node)
    return any(n in kw for n in names)


#: Registration is idempotent: importing twice must not
#: re-wrap the wrapper.
_b_registered = False


def _register() -> None:
    global _b_registered, _STOCK_COMPILE_TRANSFORM
    if _b_registered:
        return
    _b_registered = True

    # ---------------------------------------------------------------
    # Single-wedge fold fast path. PERFORMANCE ONLY.
    # ---------------------------------------------------------------
    # Stock 0.4.0 evaluates a folded child THREE times (the folded point
    # and that point rotated one sector each way, nearest wins), because
    # folding alone only answers "how far is my own copy", which
    # over-reports near a wedge boundary and lets a sphere tracer step
    # through a surface.
    #
    # That is the right default and it is wrong to remove in general.
    # It is safe HERE because of how this plate is authored: the seat
    # field is 4-fold symmetric and every pad sits strictly inside its
    # wedge (the even grid keeps each one 0.3 mm clear of the seam
    # planes), so the nearest copy to any point IS its own and the two
    # extra evaluations can only return the same distance. Measured over
    # 260k points spanning both seam planes and the wedge diagonals, on
    # the quadrant tree and the assembled tree: max |delta| 0.000000 mm,
    # zero sign flips, zero points where this path reports MORE room
    # than stock. It is worth keeping because the folded child is the
    # expensive subtree - stock costs 2.9x on the same points, on every
    # mesh, export and ray trace.
    #
    # `test_fold_fast_path_matches_stock` re-proves the equality against
    # the stock compiler on the real trees. If a pad ever crosses a
    # wedge boundary that test fails, and this block must go rather than
    # the test.
    _STOCK_COMPILE_TRANSFORM = _compile._compile_transform

    def _tf_with_static_fold(node, binding, **kw):
        if node.get("transform") == "canonical_sector_fold":
            raw_n = _compile._op_kwargs(node).get("n_sectors")
            b_plain = (
                not _op_kwargs_have(node, "centered", "phase_frac")
                and isinstance(raw_n, (int, float))
                and not isinstance(raw_n, bool)
            )
            if b_plain:
                child = _compile._compile_node(node["child"], binding, **kw)
                n_static = int(raw_n)

                def fn(p, free_vec):
                    return child(_transforms.tf_canonical_sector_fold(p, n_static), free_vec)

                return fn
        return _STOCK_COMPILE_TRANSFORM(node, binding, **kw)

    _compile._compile_transform = _tf_with_static_fold


_register()
