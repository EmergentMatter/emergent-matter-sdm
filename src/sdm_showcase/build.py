"""Primitives-showcase builder.

Primitive catalog (3 perturbations each) plus a Lofts & Sweeps showcase row.
2-D profiles are hollowed (onion) into thin ribbons and lightly extruded.
Meshes on an explicit grid (no bbox inference). Writes STLs and manifest.json
under ``mega/`` (gitignored, regenerate on demand).

Requires ``emergent-matter-sdm-core`` installed, with the loft/sweep
primitives this module's Lofts & Sweeps row needs -- see this package's
README, "Capability requirements":

    uv run python -m sdm_showcase.build [out_dir]

The geometry SDK import is deferred to :func:`_backend`, not module scope --
see that function's docstring for why.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

SHOWCASE_DIR = Path(__file__).resolve().parent

VOXEL = 0.4  # fine enough for the solids and 2-D ribbons
# TPMS walls bottom out near 0.3 mm (see the "tpms" catalog entry), and
# marching cubes needs about two voxels across a wall or it shreds it into
# disconnected flakes. So that family meshes on its own finer grid. sdm-core
# resolves each cell's sampling box from its own tree, so the fine cells cost
# what their 18 mm cubes cost and no more.
TPMS_VOXEL = 0.15
# Cells outside the TPMS family whose detail is finer than VOXEL resolves: a
# 1.4 mm thread pitch and 0.6 mm tooth at 0.4 mm shows as lumps on a print.
# They fit the trimmed TPMS box too (nothing here stands taller than ~20 mm).
FINE_CELLS = {"helix", "screw_thread", "nut_bolt_washer", "fastener_optionality"}
PX = 22.0  # perturbation x-spacing within a grid cell
# Fallback sampling box for a grid cell whose tree sdm-core cannot bound on
# its own (see mesh_tree): three perturbations across, y padded for the tilt.
GRID_HALF = [36.0, 19.0, 23.0]
sn, cs = np.sin, np.cos

# View each solid at a slight angle so its 3-D form reads (a cube looks like a
# cube, a TPMS cell looks volumetric) without moving the camera off the flat
# orthographic axis: keeping the camera dead-on is what holds the page/text
# layout square, so the *parts* tilt instead. Each primitive is rotated about
# its own centre before being placed in its row. Only the solid families tilt;
# the (flat) 2-D profiles and the lofts/sweeps band stay dead-on.
# Grid columns per family. Rows are centred by layout.py, so a family whose
# count is not a multiple of its column count ends in a shorter centred row
# rather than a ragged left-aligned one. Exact 3-D is 22 cells: six across
# gives 6/6/6/4, where five across would leave a lonely pair on the fifth row.
COLS_DEFAULT = 5
FAMILY_COLS = {"exact_3d": 6}
TILT_CATS = {"exact_3d", "tpms", "compliant"}
TILT_X = np.radians(-16.0)  # pitch: tip each top slightly toward the camera
TILT_Z = np.radians(-20.0)  # yaw: swing a side face into view (up-and-right)


def _backend() -> SimpleNamespace:
    """Import sdm-core's geometry SDK, deferred to point of use.

    STYLE.md requires heavy/host-bound dependencies to be imported lazily,
    at point of use, with a clear error telling the user how to get the
    dependency if the import fails. ``software_defined_matter`` is this
    package's one heavy dependency: importing it eagerly at module scope
    would make a plain ``import sdm_showcase.build`` fail without sdm-core
    installed, which breaks both ``verify-wheel``'s plain package import
    and ``tests/test_showcase_catalog.py``, which pins the catalog's shape
    without needing the geometry backend at all.

    Meshing goes through sdm-core's public ``export_part``: every cell is
    wrapped as a one-material ``Part`` and exported at the cell's voxel
    size, so nothing here imports a private submodule.
    """
    try:
        from software_defined_matter import MaterialRegion, Part
        from software_defined_matter import sdf_2d_to_3d as lift
        from software_defined_matter import sdf_loft as loft
        from software_defined_matter import sdf_modifier as mod
        from software_defined_matter import sdf_op as op
        from software_defined_matter import sdf_primitive as prim
        from software_defined_matter import sdf_sweep as sweep
        from software_defined_matter import sdf_transform as xform
        from software_defined_matter.export import export_part
        from software_defined_matter.grid_sampling import BBoxResolutionError
    except ImportError as exc:
        raise RuntimeError(
            "build.py needs `emergent-matter-sdm-core` installed. Run it as "
            "`uv run python -m sdm_showcase.build`, against an sdm-core "
            "checkout -- see this module's docstring."
        ) from exc
    return SimpleNamespace(
        Part=Part,
        MaterialRegion=MaterialRegion,
        export_part=export_part,
        BBoxResolutionError=BBoxResolutionError,
        P=prim,
        T=xform,
        MOD=mod,
        LIFT=lift,
        loft=loft,
        sweep=sweep,
        op=op,
    )


def tilt(sdk: SimpleNamespace, tree: Any) -> Any:
    """Apply the shared 3-D viewing tilt to a single primitive (about its own
    centre): pitch about x, then yaw about z."""
    return sdk.T("rotate_z", sdk.T("rotate_x", tree, angle=TILT_X), angle=TILT_Z)


def sc(deg: float) -> list[float]:
    a = np.radians(deg)
    return [float(sn(a)), float(cs(a))]


# -- 2-D helpers -------------------------------------------------------------
def ribbon(sdk: SimpleNamespace, t2d: Any, wall: float = 0.5, h: float = 1.6) -> Any:
    """Hollow a 2-D profile (onion -> thin band), lightly extrude, and stand it
    up facing the camera so the outline reads as a thin ribbon."""
    return sdk.T(
        "rotate_x",
        sdk.LIFT("extrusion", sdk.MOD("onion", t2d, thickness=wall), h=h),
        angle=np.pi / 2,
    )


def star(n: int, ro: float, ri: float) -> list[list[float]]:
    return [
        [
            float((ro if k % 2 == 0 else ri) * cs(np.pi * k / n)),
            float((ro if k % 2 == 0 else ri) * sn(np.pi * k / n)),
        ]
        for k in range(2 * n)
    ]


def ngon(n: int, r: float) -> list[list[float]]:
    return [[float(r * cs(2 * np.pi * k / n)), float(r * sn(2 * np.pi * k / n))] for k in range(n)]


def radial_cp(rfunc: Callable[[float], float], n: int = 22, rot: float = 0.0) -> list[list[float]]:
    th = np.linspace(0, 2 * np.pi, n, endpoint=False) + rot
    return [[float(rfunc(a) * cs(a)), float(rfunc(a) * sn(a))] for a in th]


def bez_ellipse(rx: float, ry: float) -> list[list[float]]:
    k = 0.5522847498
    return [
        [rx, 0], [rx, ry * k], [rx * k, ry], [0, ry], [-rx * k, ry], [-rx, ry * k],
        [-rx, 0], [-rx, -ry * k], [-rx * k, -ry], [0, -ry], [rx * k, -ry], [rx, -ry * k],
    ]  # fmt: skip


def superE(R: float, p: float) -> Callable[[float], float]:
    """Superellipse radius(angle): p=2 is a circle, larger p squares it off."""
    return lambda a: R / (abs(cs(a)) ** p + abs(sn(a)) ** p) ** (1.0 / p)


def circ(R: float) -> Callable[[float], float]:
    """Constant radius(angle): a plain circle."""
    return lambda a: R


def lobe(R: float, k: float, amp: float) -> Callable[[float], float]:
    """Radius(angle) with k cosine lobes of the given amplitude fraction."""
    return lambda a: R * (1.0 + amp * cs(k * a))


# -- vivid category colors (linear Base Color) -------------------------------
# CATALOG and CONSTRUCTIONS hold their builders as closures over `sdk` (bound
# in build_catalog() / build_constructions() below, once _backend() has run),
# so defining this table never touches the geometry SDK -- only invoking a
# builder does. That is what keeps a plain `import sdm_showcase.build` (and
# tests/test_showcase_catalog.py, which imports this module to pin the
# catalog's shape) safe without sdm-core installed.

# family -> (color, [(name, builder), ...])
_Family = tuple[tuple[float, float, float], list[tuple[str, Callable[[], list[Any]]]]]


# One thread spec shared by every fastener cell (root radius, tooth depth,
# pitch, all mm), and the z-extent every bolt shares: tip at FASTENER_Z_TIP,
# head seated at FASTENER_Z_HEAD. Sharing them is the point of the two cells:
# same shank, different heads, and a nut that actually fits.
FASTENER_R_ROOT, FASTENER_DEPTH, FASTENER_PITCH = 2.2, 0.6, 1.4
FASTENER_Z_TIP, FASTENER_Z_HEAD = -9.5, 6.3


def _stack(sdk: SimpleNamespace, tree: Any, z_lo: float, z_hi: float) -> Any:
    """Translate a z-centred solid so it spans ``[z_lo, z_hi]``."""
    return sdk.T("translate", tree, t=[0.0, 0.0, 0.5 * (z_lo + z_hi)])


def threaded_shank(sdk: SimpleNamespace, z_lo: float, z_hi: float) -> Any:
    """Root cylinder plus external thread ridge over ``[z_lo, z_hi]``, cut the
    way the ``screw_thread`` docstring prescribes. The band holds as many whole
    turns as fit, so the thread never pokes past the shank."""
    P = sdk.P
    length = z_hi - z_lo
    n_turns = int(length // FASTENER_PITCH)
    ridge = P(
        "screw_thread",
        r_root=FASTENER_R_ROOT,
        depth=FASTENER_DEPTH,
        pitch=FASTENER_PITCH,
        width=0.5 * FASTENER_PITCH,
        n_turns=n_turns,
    )
    shank = P("capped_cylinder", h=0.5 * length, r=FASTENER_R_ROOT)
    return _stack(sdk, sdk.op("union", [shank, ridge]), z_lo, z_hi)


def hex_socket(sdk: SimpleNamespace, z_top: float, depth: float = 1.4) -> Any:
    """A hex recess to subtract from a head whose top face is at ``z_top``:
    centred on that face so it cuts ``depth`` down and clears above."""
    return sdk.T("translate", sdk.P("hex_prism", h=[1.5, depth]), t=[0.0, 0.0, z_top])


def fasteners(sdk: SimpleNamespace) -> list[Any]:
    """Shoulder bolt, nut and washer for the ``nut_bolt_washer`` cell.

    The bolt carries a plain shoulder (a larger unthreaded land under the
    head) above its threaded length, and the nut is cut for the same thread:
    hex minus the root bore minus the external ridge. The washer is a plain
    annulus sized to the same bore.
    """
    P, T = sdk.P, sdk.T
    z_head, z_tip = FASTENER_Z_HEAD, FASTENER_Z_TIP
    z_shoulder = z_head - 5.0  # 5 mm shoulder land under the head
    bolt = sdk.op(
        "union",
        [
            _stack(sdk, P("hex_prism", h=[3.6, 1.6]), z_head, z_head + 3.2),  # hex head
            _stack(sdk, P("capped_cylinder", h=2.5, r=2.8), z_shoulder, z_head),  # shoulder
            threaded_shank(sdk, z_tip, z_shoulder),
        ],
    )
    ridge = P(
        "screw_thread",
        r_root=FASTENER_R_ROOT,
        depth=FASTENER_DEPTH,
        pitch=FASTENER_PITCH,
        width=0.5 * FASTENER_PITCH,
        n_turns=7,
    )
    nut = sdk.op(
        "subtract",
        [
            P("hex_prism", h=[4.2, 2.0]),
            P("capped_cylinder", h=3.0, r=FASTENER_R_ROOT),  # root bore
            ridge,  # internal thread = the external ridge, removed
        ],
    )
    washer = sdk.op(
        "subtract",
        [
            P("capped_cylinder", h=0.6, r=4.6),
            P("capped_cylinder", h=1.0, r=FASTENER_R_ROOT + 0.15),
        ],
    )
    face = lambda tree: T("rotate_x", tree, angle=np.pi / 2)  # noqa: E731 (axis -> camera)
    return [bolt, face(nut), face(washer)]


def fastener_optionality(sdk: SimpleNamespace) -> list[Any]:
    """Socket head cap screw, countersunk and button head, on one shank.

    Same thread, same length as the shoulder bolt next door, so the only
    thing that changes across the three slots is the head: a cylinder with a
    hex socket, a cone with a hex socket, and a spherical cap with a hex
    socket. The cut sphere keeps the cap above its cut plane, so it is
    translated to seat that plane on the head datum.
    """
    P = sdk.P
    z_head, z_tip = FASTENER_Z_HEAD, FASTENER_Z_TIP
    shank = threaded_shank(sdk, z_tip, z_head)
    head_h = 3.2
    shcs = sdk.op(
        "subtract",
        [
            sdk.op(
                "union",
                [
                    shank,
                    _stack(
                        sdk, P("capped_cylinder", h=0.5 * head_h, r=3.3), z_head, z_head + head_h
                    ),
                ],
            ),
            hex_socket(sdk, z_head + head_h),
        ],
    )
    countersunk = sdk.op(
        "subtract",
        [
            sdk.op(
                "union",
                [
                    shank,
                    _stack(
                        sdk,
                        P("capped_cone", h=0.5 * head_h, r1=FASTENER_R_ROOT, r2=4.0),
                        z_head,
                        z_head + head_h,
                    ),
                ],
            ),
            hex_socket(sdk, z_head + head_h),
        ],
    )
    r_dome, h_cut = 4.2, 1.8  # base radius sqrt(r^2 - h^2) ~= 3.8, dome height r - h = 2.4
    dome_top = z_head + (r_dome - h_cut)
    button = sdk.op(
        "subtract",
        [
            sdk.op(
                "union",
                [
                    shank,
                    sdk.T(
                        "translate",
                        P("cut_sphere", r=r_dome, h=h_cut),
                        t=[0.0, 0.0, z_head - h_cut],
                    ),
                ],
            ),
            hex_socket(sdk, dome_top, depth=1.1),
        ],
    )
    return [shcs, countersunk, button]


def build_catalog(sdk: SimpleNamespace) -> dict[str, _Family]:
    """The primitive catalog: family -> (color, [(name, builder), ...])."""
    P, T, MOD = sdk.P, sdk.T, sdk.MOD
    # fmt: off
    return {
        "exact_3d": ((1.0, 0.42, 0.04), [
            ("sphere", lambda: [P("sphere", r=r) for r in (5, 7, 9)]),
            ("box", lambda: [P("box", b=b) for b in ([6, 6, 6], [8, 5, 4], [4, 4, 9])]),
            ("round_box", lambda: [P("round_box", b=[6, 6, 6], r=r) for r in (0.5, 2, 4)]),
            ("box_frame", lambda: [P("box_frame", b=[7, 7, 7], e=e) for e in (0.4, 0.9, 1.6)]),
            ("torus", lambda: [T("rotate_x", P("torus", t=t), angle=np.pi / 2) for t in ([7, 2], [6, 3], [8, 1.5])]),
            ("capped_torus", lambda: [T("rotate_x", P("capped_torus", sc=sc(d), ra=7, rb=2.2), angle=np.pi / 2) for d in (60, 120, 150)]),
            ("link", lambda: [T("rotate_x", P("link", le=le, r1=4, r2=1.6), angle=np.pi / 2) for le in (1, 4, 7)]),
            ("cone", lambda: [P("cone", c=sc(d), h=13) for d in (18, 28, 40)]),
            ("plane", lambda: [sdk.op("intersect", [P("plane", n=n, h=0), P("box", b=[8, 8, 8])])
                               for n in ([0, 0, 1], list(np.array([0, 1, 2]) / np.sqrt(5)), [0.577, 0.577, 0.577])]),
            ("hex_prism", lambda: [P("hex_prism", h=h) for h in ([7, 7], [8, 3], [5, 11])]),
            ("tri_prism", lambda: [P("tri_prism", h=h) for h in ([7, 7], [8, 3], [5, 11])]),
            ("capsule", lambda: [P("capsule", a=[0, 0, -6], b=[0, 0, 6], r=r) for r in (2, 3, 4.5)]),
            ("capped_cylinder", lambda: [P("capped_cylinder", h=h, r=r) for h, r in ((8, 5), (4, 7), (11, 3))]),
            ("rounded_cylinder", lambda: [P("rounded_cylinder", ra=5, rb=rb, h=7) for rb in (0.5, 2, 4)]),
            ("capped_cone", lambda: [P("capped_cone", h=8, r1=r1, r2=r2) for r1, r2 in ((8, 2), (7, 5), (2, 8))]),
            ("solid_angle", lambda: [P("solid_angle", c=sc(d), ra=9) for d in (30, 50, 70)]),
            ("cut_sphere", lambda: [P("cut_sphere", r=8, h=h) for h in (-3, 0, 4)]),
            ("ellipsoid", lambda: [P("ellipsoid", r=r) for r in ([8, 5, 5], [6, 6, 10], [10, 4, 6])]),
            ("octahedron", lambda: [P("octahedron", s=s) for s in (6, 8, 10)]),
            ("pyramid", lambda: [T("scale", T("rotate_x", T("rotate_y", P("pyramid", h=h), angle=np.pi / 4), angle=-np.pi / 2 - 0.30), s=9) for h in (0.7, 1.1, 1.6)]),
            # Wound about +Z, so the tilt shows the coil as a coil rather than a stack of rings.
            ("helix", lambda: [P("helix", major_r=5, pitch=pitch, r=1.3, n_turns=3) for pitch in (3.5, 4.5, 5.5)]),
            # A thread is a ridge, not a solid: union it onto its own root cylinder (see the
            # primitive's docstring). n_turns is chosen per pitch so all three bands stand ~14 mm.
            ("screw_thread", lambda: [sdk.op("union", [P("capped_cylinder", h=8, r=4), P("screw_thread", r_root=4, depth=1.2, pitch=pitch, width=0.5 * pitch, n_turns=n)]) for pitch, n in ((2.0, 7), (2.8, 5), (3.6, 4))]),
            # Not a primitive: the three slots hold a bolt, a nut and a washer built from
            # the primitives above, to show the thread doing its job. Bolt stands on +Z;
            # nut and washer face the camera so the bore reads.
            ("nut_bolt_washer", lambda: fasteners(sdk)),
            # Same shank as the shoulder bolt, three head styles: SHCS, countersunk, button.
            ("fastener_optionality", lambda: fastener_optionality(sdk)),
        ]),
        # 18 mm cells (~= the orange row). `min_thickness` is a physical wall
        # floor in mm, normalised per family by sdm-core against that family's
        # steepest gradient, so one value fills each surface differently (a
        # neovius wall varies ~5x across one sheet, a gyroid ~1.2x). The
        # triples come from a numeric solid-fraction sweep (121^3 samples per
        # cell) that lands every family on the same ~22/40/58% ramp, floored
        # at two TPMS_VOXELs so marching cubes keeps the thinnest sheet
        # intact. Lidinoid repeats at half its nominal period, so it gets one
        # 18 mm period where the others get two 9 mm ones: same cube, same
        # visual cell count, walls twice as thick for the same fill.
        "tpms": ((0.0, 0.78, 0.70), [
            ("gyroid", lambda: [P("gyroid", period=9, min_thickness=t, n_periods=[2, 2, 2]) for t in (0.59, 1.07, 1.53)]),
            ("schwarz_p", lambda: [P("schwarz_p", period=9, min_thickness=t, n_periods=[2, 2, 2]) for t in (0.67, 1.21, 1.77)]),
            ("schwarz_d", lambda: [P("schwarz_d", period=9, min_thickness=t, n_periods=[2, 2, 2]) for t in (0.47, 0.84, 1.22)]),
            ("neovius", lambda: [P("neovius", period=9, min_thickness=t, n_periods=[2, 2, 2]) for t in (0.30, 0.44, 0.98)]),
            ("lidinoid", lambda: [P("lidinoid", period=18, min_thickness=t, n_periods=[1, 1, 1]) for t in (0.41, 0.73, 1.07)]),
        ]),
        "compliant": ((1.0, 0.10, 0.22), [
            ("notch_hinge", lambda: [P("notch_hinge", width=6, depth=5, notch_radius=r) for r in (0.8, 1.6, 2.6)]),
            ("leaf_spring", lambda: [P("leaf_spring", length=16, width=6, thickness=t) for t in (0.8, 1.6, 2.6)]),
            ("bellows", lambda: [P("bellows", outer_r=6, inner_r=4, period=3, n_periods=n) for n in (2, 3, 4)]),
            ("serpentine", lambda: [P("serpentine", amplitude=5, wavelength=8, beam_width=1.5, beam_height=3, n_periods=n) for n in (2, 3, 4)]),
            ("annular_sector", lambda: [T("rotate_x", P("annular_sector", inner_r=4, outer_r=7, half_angle=a, height=4), angle=np.pi / 2) for a in (0.6, 1.2, 2.0)]),
        ]),
        "profile_2d": ((0.05, 0.32, 1.0), [
            ("circle_2d", lambda: [ribbon(sdk, P("circle_2d", r=r)) for r in (4, 6, 8)]),
            ("box_2d", lambda: [ribbon(sdk, P("box_2d", b=b)) for b in ([6, 6], [8, 4], [4, 8])]),
            ("rounded_box_2d", lambda: [ribbon(sdk, P("rounded_box_2d", b=[6, 6], r=r)) for r in (0.5, 2, 4)]),
            ("segment_2d", lambda: [ribbon(sdk, MOD("round", P("segment_2d", a=[-6, 0], b=[6, 0]), r=r)) for r in (1, 2, 3)]),
            ("trapezoid_2d", lambda: [ribbon(sdk, P("trapezoid_2d", r1=r1, r2=r2, he=6)) for r1, r2 in ((7, 3), (5, 5), (2, 7))]),
            ("uneven_capsule_2d", lambda: [ribbon(sdk, P("uneven_capsule_2d", r1=r1, r2=r2, h=8)) for r1, r2 in ((2, 4), (4, 2), (3, 3))]),
            ("polygon_2d", lambda: [ribbon(sdk, P("polygon_2d", vertices=v)) for v in (ngon(3, 8), ngon(6, 7), star(5, 9, 4))]),
            ("bezier_2d", lambda: [ribbon(sdk, P("bezier_2d", control_points=cp)) for cp in (bez_ellipse(8, 5), bez_ellipse(6, 6), bez_ellipse(4, 9))]),
            ("bspline_2d", lambda: [ribbon(sdk, P("bspline_2d", control_points=cp)) for cp in (radial_cp(circ(7)), radial_cp(lobe(7, 4, 0.25)), star(5, 9, 5))]),
        ]),
    }
    # fmt: on


# -- Lofts & Sweeps showcase (each is one part; own bbox half) ---------------
def helix_path(r: float, turns: float, h: float, n: int = 80) -> list[list[float]]:
    return [
        [
            float(r * cs(2 * np.pi * turns * i / (n - 1))),
            float(r * sn(2 * np.pi * turns * i / (n - 1))),
            float(-h / 2 + h * i / (n - 1)),
        ]
        for i in range(n)
    ]


def circle_path_xz(radius: float, n: int = 56) -> list[list[float]]:
    return [
        [float(radius * cs(2 * np.pi * i / n)), 0.0, float(radius * sn(2 * np.pi * i / n))]
        for i in range(n)
    ]


def build_constructions(sdk: SimpleNamespace) -> list[tuple[str, list[float], Callable[[], Any]]]:
    """Lofts & sweeps: one combined row. Each entry is (name, bbox half, builder)."""
    P = sdk.P
    return [
        ("loft_morphing_column", [16, 16, 26], lambda: sdk.loft(
            [P("bspline_2d", control_points=radial_cp(superE(12, 4))),
             P("bspline_2d", control_points=radial_cp(circ(10))),
             P("bspline_2d", control_points=radial_cp(lobe(9, 5, 0.32))),
             P("bspline_2d", control_points=radial_cp(circ(7)))],
            z=[-22, -8, 8, 22], interp="shape", smooth=True)),
        ("loft_twisted_vase", [17, 17, 26], lambda: sdk.loft(
            [P("bspline_2d", control_points=radial_cp(lobe(11, 4, 0.16), rot=0.0)),
             P("bspline_2d", control_points=radial_cp(lobe(9, 4, 0.26), rot=np.pi / 8)),
             P("bspline_2d", control_points=radial_cp(lobe(8, 4, 0.32), rot=np.pi / 4)),
             P("bspline_2d", control_points=radial_cp(lobe(10, 4, 0.14), rot=3 * np.pi / 8))],
            z=[-22, -7, 8, 22], interp="shape", smooth=True)),
        ("loft_bezier", [16, 16, 24], lambda: sdk.loft(
            [P("bezier_2d", control_points=bez_ellipse(13, 6)),
             P("bezier_2d", control_points=bez_ellipse(9, 9)),
             P("bezier_2d", control_points=bez_ellipse(6, 13))],
            z=[-20, 0, 20], interp="shape", smooth=True)),
        ("sweep_ring", [16, 6, 16], lambda: sdk.sweep(
            P("circle_2d", r=2.5), circle_path_xz(13), path_kind="polyline", closed=True)),
        ("sweep_coil", [10, 10, 17], lambda: sdk.sweep(
            P("circle_2d", r=1.7), helix_path(7, 3.0, 28), path_kind="polyline", closed=False)),
        ("sweep_bspline_path", [18, 6, 18], lambda: sdk.sweep(
            P("bspline_2d", control_points=radial_cp(superE(2.6, 4))),
            [[float(radius * cs(2 * np.pi * i / 6)), 0.0, float(radius * sn(2 * np.pi * i / 6))]
             for i, radius in enumerate([16, 8, 16, 8, 16, 8])],
            path_kind="bspline", closed=True)),
    ]  # fmt: skip


def mesh_tree(
    sdk: SimpleNamespace,
    tree: Any,
    path: Path,
    voxel: float = VOXEL,
    half: list[float] | None = None,
) -> None:
    """Mesh one cell's SDF tree to ``path`` as STL, through sdm-core's public exporter.

    The tree is wrapped as a one-material ``Part``; ``export_part`` resolves
    the sampling box from the tree, samples it at ``voxel``, runs marching
    cubes and the conservative cleanup, and writes the file. It names files
    after the part and material, so it exports into a scratch directory
    beside ``path`` and the result is moved into place.

    sdm-core's analytic bounder declines some trees (an extrusion whose child
    is a modifier, for one, which is every 2-D profile ribbon here). Its
    documented escape hatch is an explicit ``metadata["bbox"]``; when the
    caller supplies ``half`` and inference fails, the export is retried with
    that box. Inference is tried first because it gives each cell a tight box
    where it works, which is what keeps the fine-voxel cells cheap.
    """
    materials = [sdk.MaterialRegion(material_id=1, name="m", sdf_tree=tree)]
    part = sdk.Part(name="cell", params={}, materials=materials)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        try:
            (written,) = sdk.export_part(part, tmp, voxel_size=voxel, fmt="stl", stamp=False)
        except sdk.BBoxResolutionError:
            if half is None:
                raise
            box = [[-float(h) for h in half], [float(h) for h in half]]
            part = sdk.Part(name="cell", params={}, materials=materials, metadata={"bbox": box})
            (written,) = sdk.export_part(part, tmp, voxel_size=voxel, fmt="stl", stamp=False)
        shutil.move(written, path)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    sdk = _backend()
    out = Path(argv[0]).resolve() if argv else SHOWCASE_DIR / "mega"
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    row_off = 0
    for cat, (color, items) in build_catalog(sdk).items():
        cols = FAMILY_COLS.get(cat, COLS_DEFAULT)
        for i, (name, builder) in enumerate(items):
            row, col = row_off + i // cols, i % cols
            stl = out / f"{name}.stl"
            try:
                parts = builder()
                if cat in TILT_CATS and name != "pyramid":  # pyramid carries its own 3/4 view
                    parts = [tilt(sdk, t) for t in parts]
                trees = [
                    sdk.T("translate", t, t=[float((k - 1) * PX), 0.0, 0.0])
                    for k, t in enumerate(parts)
                ]
                voxel = TPMS_VOXEL if (cat == "tpms" or name in FINE_CELLS) else VOXEL
                mesh_tree(sdk, sdk.op("union", trees), stl, voxel, half=GRID_HALF)
                manifest.append(
                    {
                        "name": name,
                        "section": "grid",
                        "category": cat,
                        "color": list(color),
                        "stl": str(stl.relative_to(SHOWCASE_DIR, walk_up=True)),
                        "row": row,
                        "col": col,
                    }
                )
                print(f"OK  grid  {cat:11s} {name}")
            except Exception as exc:
                # Deliberately broad: one primitive's meshing failure must not
                # abort the whole batch build, only skip that one STL.
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        row_off += (len(items) + cols - 1) // cols

    purple = (0.62, 0.12, 0.95)
    for idx, (name, half, builder) in enumerate(build_constructions(sdk)):
        stl = out / f"{name}.stl"
        try:
            mesh_tree(sdk, builder(), stl, half=half)
            manifest.append(
                {
                    "name": name,
                    "section": "build",
                    "category": "construction",
                    "color": list(purple),
                    "stl": str(stl.relative_to(SHOWCASE_DIR, walk_up=True)),
                    "row": idx // 3,
                    "col": idx % 3,
                    "half": half,
                }
            )
            print(f"OK  build {name}")
        except Exception as exc:
            # Same reasoning as the grid loop above: skip, don't abort.
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)} cells -> {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
