"""SDF geometry for the disco-lens plate, built from sdm-core primitives.

The part is one MaterialRegion:

    subtract(
        union(square base slab, hub, fold_4(one quadrant of mirror seats)),
        union(stick hole, everything below the plate underside),
    )

A mirror seat is an OPEN tilted pad (no pocket - Rob's material/density
call) plus an L-shaped locating corner: two low stubs the mirror is pushed
against while the glue sets. The stub of one site is the only thing
between neighboring mirrors, so the grid pitch is mirror + clearances +
one stub.

The centered even grid is exactly 4-fold rotationally symmetric, so only
the first-quadrant seats are authored and a single ``canonical_sector_fold
(n=4)`` replicates them - the same O(1)-evaluation trick the ring design
used, which keeps meshing ~4x cheaper than authoring every seat. The L
corner therefore rotates 90 deg from quadrant to quadrant (assembly: push
the mirror into whichever corner has walls).

Each pad tilts toward the center about its local tangential axis while its
edges stay along the plate axes: the rotation is the conjugation
Rz(az) * Ry(-tilt) * Rz(-az). Pads root a couple of mm into the slab; on
steeper sites the downhill root corner pokes below the underside and a
plane cutter trims the bottom dead flat.

Transform sign conventions (verified in tests/test_assembly.py):
sdm-core transforms act on the QUERY point, so ``translate(t)`` moves
geometry by +t, while ``rotate_*(angle=a)`` rotates geometry by -a about
that axis. The helpers below take geometry-space arguments and flip signs
internally so nobody else has to think about this again.
"""

from __future__ import annotations

import math
from datetime import date

from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    sdf_op,
    sdf_primitive,
    sdf_transform,
)

from disco_lens.optics import Site, pad_top_z
from disco_lens.parameters import DiscoLensParameters, raise_if_invalid

# Pads root this far below the plate top face so they fuse with the slab.
D_PAD_EMBED = 2.5


def _box(d_hx: float, d_hy: float, d_hz: float) -> dict:
    return sdf_primitive("box", b=[d_hx, d_hy, d_hz])


def _rot_y_geometry(child: dict, d_angle: float) -> dict:
    """Rotate GEOMETRY by +d_angle about Y (query rotated by -d_angle)."""
    return sdf_transform("rotate_y", child, angle=-d_angle)


def _rot_z_geometry(child: dict, d_angle: float) -> dict:
    return sdf_transform("rotate_z", child, angle=-d_angle)


def _rot_x_geometry(child: dict, d_angle: float) -> dict:
    return sdf_transform("rotate_x", child, angle=-d_angle)


def _translate(child: dict, t: list[float]) -> dict:
    return sdf_transform("translate", child, t=t)


def _box_span(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> dict:
    """Axis-aligned box by min/max spans (authoring convenience)."""
    return _translate(
        _box((x1 - x0) / 2.0, (y1 - y0) / 2.0, (z1 - z0) / 2.0),
        [(x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0],
    )


def _rounded_slab(d_hx: float, d_hy: float, d_r: float, z0: float, z1: float) -> dict:
    """Centered slab with rounded plan corners, built from two boxes and
    four corner cylinders."""
    d_hz = (z1 - z0) / 2.0
    d_zc = (z0 + z1) / 2.0
    corners = [
        _translate(
            sdf_primitive("capped_cylinder", h=d_hz, r=d_r),
            [sx * (d_hx - d_r), sy * (d_hy - d_r), d_zc],
        )
        for sx in (1.0, -1.0)
        for sy in (1.0, -1.0)
    ]
    return sdf_op(
        "union",
        [
            _box_span(-d_hx + d_r, d_hx - d_r, -d_hy, d_hy, z0, z1),
            _box_span(-d_hx, d_hx, -d_hy + d_r, d_hy - d_r, z0, z1),
            *corners,
        ],
    )


def _seat_unit(p: DiscoLensParameters, site: Site) -> dict:
    """One mirror seat, world-placed: a bare tilted pad. The mirror is
    glued flat onto it - no locating walls (they cost material and were
    measured to block ~4% of the reflected light). The pad outline, a
    0.2 mm ledge around the mirror, is the placement guide.

    Authored in a local frame with the PAD TOP CENTER at the origin, then
    conjugate-tilted toward the plate center and translated to the site.
    """
    d_reach = p.d_pad_reach
    d_top = pad_top_z(p, site, site.d_tilt)
    d_hz = (d_top + D_PAD_EMBED) / (2.0 * math.cos(site.d_tilt))

    unit = _box_span(-d_reach, d_reach, -d_reach, d_reach, -2.0 * d_hz, 0.0)

    d_az = site.d_azimuth
    tilted = _rot_z_geometry(_rot_y_geometry(_rot_z_geometry(unit, -d_az), -site.d_tilt), d_az)
    pad = _translate(tilted, [site.d_x, site.d_y, d_top])

    # FUSION COLLAR: a WORLD-AXIS box around the pad column, d_fuse_ledge
    # wider than the mirror, with a HORIZONTAL top d_fuse_drop below the
    # pad-plane center. Collars overlap the neighbors' so the field
    # prints as one body. The top is deliberately horizontal, not tilted
    # with the pad: tilted collar tops crossed the slab top face and
    # their fold-twins at the seams at grazing angles - knife lines that
    # pinched the mesh non-manifold. Horizontal tops are parallel to the
    # slab (never intersect), meet their seam twins at identical heights
    # (one continuous plane), and step against neighbors with vertical
    # walls - every junction is parallel or perpendicular. They also
    # bounce stray sun straight up instead of spraying it sideways.
    d_ct = collar_top_z(p, site)
    if d_ct < 0.4:
        return pad
    d_rc = p.d_mirror_half + p.d_fuse_ledge
    collar = _box_span(
        site.d_x - d_rc,
        site.d_x + d_rc,
        site.d_y - d_rc,
        site.d_y + d_rc,
        -p.d_base_thickness + 0.5,
        d_ct,
    )
    return sdf_op("union", [pad, collar])


def collar_top_z(p: DiscoLensParameters, site: Site) -> float:
    """World z of a seat's fusion-collar top.

    Three rules, each earned the hard way:
    - CAPPED below the pad's LOWEST corner: a horizontal top referenced
      to the pad-plane CENTER stood ~2.7 mm proud of steep corner pads'
      downhill edges - Rob spotted the shelf, and its lip actually rose
      past the glass's reflecting surface and ate the bottom slice of
      the mirror's own beam.
    - QUANTIZED to a 0.7 grid (2 voxels): overlapping neighbor tops are
      either exactly coplanar (one clean surface) or step >= 2 voxels -
      never a sub-voxel razor shelf (pinched v1's mesh).
    - Callers SKIP the collar below 0.4: the slab fuses everything under
      z=0, and a top within a voxel of the slab face is coplanar poison.
    """
    d_lowest = p.d_pad_clearance + p.pad_raise(site.d_radius)
    d_ct = min(pad_top_z(p, site, site.d_tilt) - p.d_fuse_drop, d_lowest - 0.2)
    return math.floor(d_ct / 0.7) * 0.7


def _glass_keepout(p: DiscoLensParameters, site: Site) -> dict:
    """Cutter guaranteeing NOTHING exists inside a seat's glass envelope:
    a tilted box over the mirror footprint, from a hair below the pad
    plane to well above the field. Every cut face is deliberately
    TRANSVERSE: the walls sit 0.1 OUTSIDE the pad's own edge planes and
    the floor 0.05 INSIDE the pad top (coincident walls/floor shattered
    marching cubes on every flagged seat - 'STILL LEAKY'). Flagged seats
    end up 0.05 lower; the optics model's glue-line tolerance dwarfs it."""
    d_r = p.d_pad_reach + 0.1
    unit = _box_span(-d_r, d_r, -d_r, d_r, -0.05, 40.0)
    d_az = site.d_azimuth
    tilted = _rot_z_geometry(_rot_y_geometry(_rot_z_geometry(unit, -d_az), -site.d_tilt), d_az)
    return _translate(tilted, [site.d_x, site.d_y, pad_top_z(p, site, site.d_tilt)])


def _seat_field(
    p: DiscoLensParameters, all_sites: list[Site], authored: list[Site]
) -> tuple[dict, dict | None]:
    """(seat union, glass keep-out cutter) - the cutter must be
    subtracted from the WHOLE solid, not just the seats: at the plate
    corners the stiffening RIM's rounded arc crosses the corner seats'
    glass envelopes too (one of the 12 clashes on the first plate
    printed was the rim, not a pad). ``authored`` is the subset
    actually emitted (Q1 only); flags always come from the FULL site
    list."""
    # EVERY seat gets a keep-out cutter, unconditionally. A model that
    # predicted which seats needed one missed three different ways at
    # three different scales (rim arcs, tilted-frame corner lean, the
    # on-edge revision's steeper corners) - the guarantee is structural
    # now, not modeled.
    units = [_seat_unit(p, s) for s in authored]
    seats = sdf_op("union", units)
    cutter_list = [_glass_keepout(p, s) for s in authored]
    cutters = sdf_op("union", cutter_list) if cutter_list else None
    seats = sdf_transform("canonical_sector_fold", seats, n_sectors=4)
    if cutters is not None:
        cutters = sdf_transform("canonical_sector_fold", cutters, n_sectors=4)
    return seats, cutters


# =========================================================================
# The plate: four glued quadrant plates + pedestal clamp ring
# =========================================================================


def _sight_tab(p) -> tuple[dict, dict, dict, dict]:
    """The Q1 sight tab on the +X edge, centered at y = d_tab_center_y:
    (tab block, boss+fillet, pinhole cutter, footprint strip). The other
    three quadrants get rotated copies - the tab is part of the fold
    orbit, one per quadrant."""
    d_cy = p.d_tab_center_y
    d_hw = p.d_sight_tab_width / 2.0
    d_rt = p.d_sight_tab_corner_radius
    d_x1 = p.d_plate_half + p.d_sight_tab_length

    tab_parts = []
    for z0, z1, d_x_in in [
        (-p.d_base_thickness, 0.0, p.d_plate_half - p.d_sight_tab_root),
        (0.0, p.d_rim_height, p.d_plate_half - p.d_rim_thickness),
    ]:
        d_hz = (z1 - z0) / 2.0
        tab_parts.append(
            sdf_op(
                "union",
                [
                    _box_span(d_x_in, d_x1 - d_rt, d_cy - d_hw, d_cy + d_hw, z0, z1),
                    _box_span(d_x_in, d_x1, d_cy - d_hw + d_rt, d_cy + d_hw - d_rt, z0, z1),
                    *[
                        _translate(
                            sdf_primitive("capped_cylinder", h=d_hz, r=d_rt),
                            [d_x1 - d_rt, d_cy + s * (d_hw - d_rt), (z0 + z1) / 2.0],
                        )
                        for s in (1.0, -1.0)
                    ],
                ],
            )
        )
    tab = sdf_op("union", tab_parts)

    d_cx = p.d_sight_center_x
    d_fr = p.d_sight_boss_fillet
    boss = sdf_op(
        "union",
        [
            _translate(
                sdf_primitive(
                    "capped_cylinder",
                    h=p.d_sight_boss_height / 2.0,
                    r=p.d_sight_boss_diameter / 2.0,
                ),
                [d_cx, d_cy, p.d_sight_boss_height / 2.0],
            ),
            _translate(
                sdf_primitive(
                    "capped_cone",
                    h=d_fr / 2.0,
                    r1=p.d_sight_boss_diameter / 2.0 + d_fr,
                    r2=p.d_sight_boss_diameter / 2.0,
                ),
                [d_cx, d_cy, p.d_rim_height + d_fr / 2.0],
            ),
        ],
    )
    hole = _translate(
        sdf_primitive(
            "capped_cylinder",
            h=p.d_sight_boss_height / 2.0 + p.d_base_thickness + 2.0,
            r=p.d_sight_hole_diameter / 2.0,
        ),
        [d_cx, d_cy, p.d_sight_boss_height / 2.0],
    )
    d_hclip = max(p.d_sight_boss_height, p.d_hub_height) + 10.0
    strip = _box_span(0.0, d_x1, d_cy - d_hw, d_cy + d_hw, -d_hclip, d_hclip)
    return tab, boss, hole, strip


def build_tree(p, sites: list[Site], b_quadrant: bool = True) -> dict:
    """The plate. b_quadrant=True cuts the PRINTED PART: the first
    quadrant (x >= 0, y >= 0) with its quarter of the pedestal, one seam
    face along each fold plane, and its one sight tab. b_quadrant=False
    is the full glued-up assembly (for ray tracing and sanity checks)."""
    d_r_hub = p.d_hub_diameter / 2.0
    d_h_hub = (p.d_hub_height + p.d_base_thickness) / 2.0
    d_hp = p.d_plate_half
    d_rc = p.d_plate_corner_radius

    base = _rounded_slab(d_hp, d_hp, d_rc, -p.d_base_thickness, 0.0)
    rim = sdf_op(
        "subtract",
        [
            _rounded_slab(d_hp, d_hp, d_rc, 0.0, p.d_rim_height),
            _rounded_slab(
                d_hp - p.d_rim_thickness,
                d_hp - p.d_rim_thickness,
                max(d_rc - p.d_rim_thickness, 1.0),
                -1.0,
                p.d_rim_height + 1.0,
            ),
        ],
    )
    hub = _translate(
        sdf_primitive("capped_cylinder", h=d_h_hub, r=d_r_hub),
        [0.0, 0.0, p.d_hub_height - d_h_hub],
    )
    hub_skirt = _translate(
        sdf_primitive("capped_cone", h=2.0, r1=d_r_hub + 4.0, r2=d_r_hub), [0.0, 0.0, 2.0]
    )

    authored = [s for s in sites if s.d_x > 0.0 and s.d_y > 0.0]
    seats, glass_cutters = _seat_field(p, sites, authored)

    tab1, boss1, hole1, strip1 = _sight_tab(p)
    n_copies = 1 if b_quadrant else 4
    tabs = [_rot_z_geometry(tab1, k * math.pi / 2.0) for k in range(n_copies)]
    bosses = [_rot_z_geometry(boss1, k * math.pi / 2.0) for k in range(n_copies)]
    sight_holes = [_rot_z_geometry(hole1, k * math.pi / 2.0) for k in range(n_copies)]
    strips = [_rot_z_geometry(strip1, k * math.pi / 2.0) for k in range(n_copies)]

    solid = sdf_op("union", [base, rim, hub, hub_skirt, seats, *tabs, *bosses])

    stick_hole = sdf_primitive(
        "capped_cylinder",
        h=p.d_hub_height + p.d_base_thickness + 2.0,
        r=p.d_stick_hole_diameter / 2.0,
    )
    d_big = p.d_plate_size
    underside_trim = _box_span(
        -d_big, d_big, -d_big, d_big, -p.d_base_thickness - 30.0, -p.d_base_thickness
    )
    d_hclip = max(p.d_sight_boss_height, p.d_hub_height) + 10.0
    footprint = sdf_op(
        "union",
        [
            _rounded_slab(d_hp, d_hp, d_rc, -d_hclip, d_hclip),
            *strips,
        ],
    )
    hole_list = [stick_hole, underside_trim, *sight_holes]
    if glass_cutters is not None:
        hole_list.append(glass_cutters)
    tree = sdf_op(
        "subtract",
        [
            sdf_op("intersect", [solid, footprint]),
            sdf_op("union", hole_list),
        ],
    )
    if b_quadrant:
        # The printed part: everything past the seam planes. The cut is
        # INSET 0.05 from the fold planes - cutting at exactly x=0/y=0
        # slices along the sector-fold boundary, where mirrored collar
        # surfaces run tangent to the cut and pinch the mesh non-manifold
        # (measured: 2 pinches, both exactly on the seams). Assembled
        # plate is 410.3 with a 0.1 glue-filled gap per seam. Fused pads
        # still overhang and get trimmed flat; the glued assembly fuses
        # pads across the joint.
        tree = sdf_op("intersect", [tree, _box_span(0.05, d_big, 0.05, d_big, -d_big, d_big)])
    return tree


def build_ring_tree(p) -> dict:
    """The pedestal clamp ring: a plain sleeve that slides down over the
    four assembled quarter-posts and holds them together; the dog seats
    on the post+ring top. Bore chamfered at the bottom for lead-in. All
    cutters overrun the faces they pierce (transverse-junction rule)."""
    d_h = p.d_ring_height
    ring = _translate(
        sdf_primitive("capped_cylinder", h=d_h / 2.0, r=p.d_ring_outer_radius),
        [0.0, 0.0, d_h / 2.0],
    )
    bore = _translate(
        sdf_primitive("capped_cylinder", h=d_h / 2.0 + 1.0, r=p.d_ring_bore_radius),
        [0.0, 0.0, d_h / 2.0],
    )
    chamfer = _translate(
        sdf_primitive(
            "capped_cone", h=1.25, r1=p.d_ring_bore_radius + 0.8, r2=p.d_ring_bore_radius
        ),
        [0.0, 0.0, 0.75],
    )  # spans z = -0.5 .. 2.0, pokes out the bottom
    return sdf_op("subtract", [ring, bore, chamfer])


#: The declared params this CEM ships in the quadrant .sdm, as
#: (dataclass field, name it emits under). Every entry must be a
#: DECLARED field: a derived value shipped here becomes a second source
#: of truth that nothing keeps in sync, which is how a param edit
#: produces geometry that violates the model with nothing reporting a
#: problem. cem.toml carries the same mapping as `emits`, and
#: test_manifest.py checks the two against each other.
#: Bounds ride along because the DOCUMENT needs them, not just the
#: manifest: a param shipped without them renders in a viewer as a
#: control nothing can move. test_manifest.py checks these against
#: cem.toml so the two cannot drift.
PARAM_FIELDS = [
    ("d_plate_size", "assembled_plate_size", (100.0, 600.0)),
    ("d_base_thickness", "base_thickness", (1.5, 10.0)),
    ("d_mirror_size", "mirror_size", (5.0, 50.0)),
    ("d_stick_hole_diameter", "stick_hole_diameter", (2.0, 12.0)),
    ("d_dog_length", "dog_length", (50.0, 300.0)),
]


def build_quadrant_part(p, sites: list[Site]) -> Part:
    raise_if_invalid(p)
    part = Part(
        name="disco_lens_quadrant",
        metadata={
            "author": "EmergentMatter / Rob",
            "date": date.today().isoformat(),
            "units": "mm",
            "am_process": "FDM",
            "notes": (
                "ONE of FOUR identical quadrant plates - print four, glue "
                "the seam edges into a "
                f"{p.d_plate_size:.0f} mm square. Each carries a quarter "
                "of the center pedestal; the clamp ring "
                "(disco_lens_ring) slides over the assembled post. "
                "Print flat side down, no supports."
            ),
            "n_mirrors_per_quadrant": len(sites) // 4,
            "n_mirrors_assembled": len(sites),
            # Derived facts live here, not in params: a reader wants
            # them, but a manifest declares surface and these all follow
            # from the params above.
            "n_grid_cells": p.n_grid_cells,
            "grid_pitch_mm": p.d_grid_pitch,
            "quadrant_size_mm": p.d_quadrant_size,
            "dog_bottom_height_mm": p.d_dog_bottom_height,
        },
    )
    for s_field, s_emits, bounds in PARAM_FIELDS:
        part.add_param(Param(s_emits, getattr(p, s_field), free=False, bounds=bounds, unit="mm"))
    part.add_material(MaterialRegion(material_id=1, name="petg_fdm", sdf_tree=build_tree(p, sites)))
    return part


def build_ring_part(p) -> Part:
    raise_if_invalid(p)
    part = Part(
        name="disco_lens_ring",
        metadata={
            "units": "mm",
            "am_process": "FDM",
            "notes": (
                "Pedestal clamp ring: slides over the four glued "
                "quarter-posts, flush with the post top; the dog "
                "seats on the combined top face. Chamfered bore "
                "goes DOWN. Ream lightly if the fit is tight."
            ),
        },
    )
    part.add_material(MaterialRegion(material_id=1, name="petg_fdm", sdf_tree=build_ring_tree(p)))
    return part
