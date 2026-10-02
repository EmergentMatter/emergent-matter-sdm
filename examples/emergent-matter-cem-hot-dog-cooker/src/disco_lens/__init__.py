"""disco-lens: FDM-printed solar hot-dog cooker mirror plate."""

import disco_lens.sdm_ext  # noqa: F401  (registers the fold fast path)
from disco_lens.optics import Design, Site, design
from disco_lens.parameters import DiscoLensParameters
from disco_lens.sdf_assembly import build_quadrant_part, build_ring_part

__all__ = [
    "Design",
    "DiscoLensParameters",
    "Site",
    "build_quadrant_part",
    "build_ring_part",
    "design",
]
