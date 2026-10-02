"""Build and verify a clamp-on backpack hook for a table edge.

The part is a "C" that wraps the table edge: a thin tapered bar rests on the
table top, a spine drops down the edge, a jaw runs back under the table, and
from the jaw's inner end the profile loops back toward the edge to form a
rounded hook under the table. The backpack's top handle slides in from the
front, over the upturned lip, and sits on the lower leg.

Frame (millimeters). The profile is drawn in XY exactly as it is printed:
  +X  outward, away from the table (the table edge face is x = 0)
  +Y  up when installed (the table top surface is y = 0)
  +Z  the hook's width along the table edge = print height, z in [0, width]
Print it lying on its side (z = 0 on the bed). Every bending stress then runs
along the layers, not across them, and the extruded profile needs no support.

Every dimension is a $ref expression over the part's params, so the .sdm can be
re-sized (for example a different table thickness) without re-running this.
Run from this directory:
  uv run python -B build.py
"""

import copy
import json
import logging
import math
from pathlib import Path

import numpy as np
import trimesh
from emergent_matter_materials import MATERIALS
from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    load,
    save,
    sdf_2d_to_3d,
    sdf_modifier,
    sdf_primitive,
    sdf_transform,
)
from software_defined_matter.export import DecimateConfig, export_part
from software_defined_matter.grid_sampling import bind_sdf
from software_defined_matter.grid_sampling.bind import resolve_bbox

OUT = Path(__file__).resolve().parent
D_VOXEL_MM = 0.2
D_G = 9.81
# Segments per 90 degrees of arc in the outline polygon. At 24, the chord error
# on the largest arc (the 24 mm outer loop radius) is 0.013 mm.
N_ARC_PER_90 = 24
# Measured edge round on the desk the hook is built for (2026-09-28).
D_DESK_EDGE_RADIUS_MM = 3.25


class E:
    """Thin wrapper that turns Python arithmetic into sdm-core DSL expressions."""

    def __init__(self, node):
        self.node = node

    @staticmethod
    def of(value):
        if isinstance(value, E):
            return value.node
        return {"type": "num", "value": float(value)}

    def _bin(self, op, other, b_swap=False):
        lhs, rhs = E.of(self), E.of(other)
        if b_swap:
            lhs, rhs = rhs, lhs
        return E({"type": "binop", "op": op, "lhs": lhs, "rhs": rhs})

    def __add__(self, o):
        return self._bin("+", o)

    def __radd__(self, o):
        return self._bin("+", o, True)

    def __sub__(self, o):
        return self._bin("-", o)

    def __rsub__(self, o):
        return self._bin("-", o, True)

    def __mul__(self, o):
        return self._bin("*", o)

    def __rmul__(self, o):
        return self._bin("*", o, True)

    def __truediv__(self, o):
        return self._bin("/", o)

    def __rtruediv__(self, o):
        return self._bin("/", o, True)

    def __neg__(self):
        return E({"type": "unop", "op": "neg", "child": self.node})

    def sqrt(self):
        return E({"type": "unop", "op": "sqrt", "child": self.node})


def leaf(value):
    return E.of(value) if isinstance(value, E) else float(value)


# fmt: off
PARAMS = [
    # name,               value, bounds,        group,    meaning
    ("d_table_thickness", 25.4, (10.0, 50.0),  "table",  "Measured table-top thickness"),
    ("d_table_clearance",  0.6, (0.0, 3.0),    "table",  "Slip-fit gap added to the table thickness"),
    ("d_width",           45.0, (20.0, 80.0),  "body",   "Width along the table edge (print height)"),
    ("d_wall",            12.0, (6.0, 16.0),   "body",   "Section thickness of the bent bar; also the top-bar root"),
    ("d_top_tip",          3.0, (2.0, 8.0),    "top",    "Top-bar thickness at its inner tip"),
    ("d_top_length",      70.0, (30.0, 150.0), "top",    "Top-bar reach onto the table from the edge"),
    ("d_jaw_depth",       30.0, (15.0, 100.0), "hook",   "Jaw reach under the table, to the loop-back"),
    ("d_throat",          24.0, (12.0, 50.0),  "hook",   "Inside height of the hook for the handle"),
    ("d_lip_height",      10.0, (3.0, 25.0),   "hook",   "Lip rise above the lower leg (retains the handle)"),
    ("d_lip_setback",      2.0, (0.0, 20.0),   "hook",   "Lip outer face distance behind the table edge"),
    ("d_edge_radius",      1.0, (0.2, 3.0),    "finish", "Rounding on the side-face edges and convex corners"),
    ("d_fillet_table",     5.0, (1.0, 10.0),   "finish", "Inside bend radius at the two corners that hug the table"),
    ("d_fillet_hook",      3.0, (0.5, 8.0),    "finish", "Inside bend radius where the lip rises from the leg"),
]
# fmt: on
DEFAULTS = {s_name: d_value for s_name, d_value, *_ in PARAMS}
P = {s_name: E({"type": "param", "name": s_name}) for s_name in DEFAULTS}


def arc(cx, cy, r, d_deg0, d_deg1):
    """Outline vertices along an arc, both ends included, angles in degrees."""
    n_seg = max(2, round(abs(d_deg1 - d_deg0) / 90 * N_ARC_PER_90))
    pts = []
    for n_i in range(n_seg + 1):
        d_a = math.radians(d_deg0 + (d_deg1 - d_deg0) * n_i / n_seg)
        pts.append((cx + r * math.cos(d_a), cy + r * math.sin(d_a)))
    return pts


def outline(d_in=0.0):
    """Closed profile outline, inset by d_in: one bent bar, every corner a true arc.

    The profile is a single polygon, not a union of pieces, so its distance
    field is exact inside and out. Pieces butted together hide internal faces
    that read too shallow to the side-edge rounding, which dented every seam.

    Every arc is concentric, so the inset outline is the same construction with
    convex radii shrunk by d_in, concave radii grown by d_in and flat faces
    moved in by d_in.
    """
    d_tc = P["d_table_thickness"] + P["d_table_clearance"]  # jaw-to-top-bar gap
    wall, gap = P["d_wall"], P["d_throat"]
    rf, rh = P["d_fillet_table"], P["d_fillet_hook"]
    # The spine stands one bend radius (plus slip) off the table edge face, so
    # each table-side bend is tangent to the table exactly at its edge: any
    # table edge fits, even a knife-sharp one, while the bend stays generous.
    c = P["d_table_clearance"]
    c1 = (c, -rf)  # top bend center
    c2 = (c, -d_tc + rf)  # bottom bend center
    cx, cy = -P["d_jaw_depth"], -d_tc - wall - gap / 2  # loop-back center
    r_loop_in, r_loop_out = gap / 2, gap / 2 + wall
    y_leg_bot = cy - r_loop_out
    y_leg_top = y_leg_bot + wall
    x_lip_out = -P["d_lip_setback"]
    c3 = (x_lip_out - wall - rh, y_leg_top + rh)  # lip bend center
    y_lip_top = y_leg_top + P["d_lip_height"]
    c4 = (x_lip_out - wall / 2, y_lip_top - wall / 2)  # lip cap center
    # Tapered top face, offset down its own normal (it leaves the top bend
    # within 7 degrees of tangent, so the joint vertex shifts < 0.01 mm).
    d_slope = (wall - P["d_top_tip"]) / (c + P["d_top_length"])
    x_tip = -P["d_top_length"] + d_in
    y_tip = wall + d_slope * (x_tip - c) - d_in * (1 + d_slope * d_slope).sqrt()
    return [
        (x_tip, d_in),
        *arc(*c1, rf + d_in, 90, 0),  # under the top bar, round the table's top edge
        *arc(*c2, rf + d_in, 0, -90),  # down the edge face, round the table's bottom edge
        *arc(cx, cy, r_loop_out - d_in, 90, 270),  # jaw top, then the outside of the loop
        *arc(*c3, rh + wall - d_in, -90, 0),  # leg bottom, front-bottom corner
        *arc(*c4, wall / 2 - d_in, 0, 180),  # up the lip, over its rounded cap
        *arc(*c3, rh + d_in, 0, -90),  # down the lip's inside, into the leg
        *arc(cx, cy, r_loop_in + d_in, 270, 90),  # leg top, inside of the loop
        *arc(*c2, rf + wall - d_in, -90, 0),  # jaw bottom, outside of the lower bend
        *arc(*c1, rf + wall - d_in, 0, 90),  # spine outside, outside of the upper bend
        (x_tip, y_tip),  # tapered top face back to the tip
    ]


def build_tree():
    r = P["d_edge_radius"]
    # Rounded extrusion: the profile inset by r, extruded short of the side
    # faces by r, then dilated by r. The polygon field is exact, so the
    # dilation puts a true radius-r round on every side edge and convex corner.
    verts = [[leaf(x), leaf(y)] for x, y in outline(r)]
    core = sdf_2d_to_3d(
        "extrusion", sdf_primitive("polygon_2d", vertices=verts), h=leaf(P["d_width"] / 2 - r)
    )
    body = sdf_modifier("round", core, r=leaf(r))
    return sdf_transform("translate", body, t=[0.0, 0.0, leaf(P["d_width"] / 2)])


def winkler_inner(d_m, d_b, d_t, d_ri):
    """Inner-fibre stress of a rectangular curved beam (Winkler), N and mm."""
    d_ro = d_ri + d_t
    d_rn = d_t / math.log(d_ro / d_ri)
    d_e = d_ri + d_t / 2 - d_rn
    return d_m * (d_rn - d_ri) / (d_b * d_t * d_e * d_ri)


def bending_report(d):
    """Hand-calc stress for a sustained load, worst handle position, per section."""
    d_yield_mpa = MATERIALS["pla_3dprint"].structural.yield_stress.d_value / 1e6
    d_b, d_t = d["d_width"], d["d_wall"]
    x_spine = d["d_table_clearance"] + d["d_fillet_table"] + d_t / 2
    # Handle pushed to the back of the loop: largest lever on the table bends.
    d_arm_bend = x_spine + d["d_jaw_depth"] + d["d_throat"] / 2
    # Handle resting against the lip: largest lever on the loop.
    d_arm_loop = d["d_jaw_depth"] - d["d_lip_setback"] - d_t
    rows = {}
    for n_kg in (10, 15, 20):
        d_w = n_kg * D_G
        s_bend = winkler_inner(d_w * d_arm_bend, d_b, d_t, d["d_fillet_table"])
        s_loop = winkler_inner(d_w * d_arm_loop, d_b, d_t, d["d_throat"] / 2)
        s_straight = 6 * d_w * d_arm_bend / (d_b * d_t**2)
        rows[f"{n_kg}_kg"] = {
            "table_bends_mpa": round(s_bend, 2),
            "loop_mpa": round(s_loop, 2),
            "straight_top_bar_root_mpa": round(s_straight, 2),
            "safety_factor_vs_yield": round(d_yield_mpa / max(s_bend, s_loop), 2),
        }
    return {
        "method": "M = W * arm; Winkler curved-beam inner-fibre stress at each bend",
        "pla_yield_mpa": d_yield_mpa,
        "arm_table_bends_mm": round(d_arm_bend, 2),
        "arm_loop_mm": d_arm_loop,
        "loads": rows,
        "sustained_creep_guide_mpa": round(0.25 * d_yield_mpa, 1),
        # The pad must reach past the load line or the clip pries on the edge.
        "load_line_inside_top_bar": d_arm_bend - x_spine < d["d_top_length"],
    }


def build_part():
    """The hook as an sdm-core Part at the default params; cem.toml's entry_point."""
    params = {
        s_name: Param(
            s_name, d_value, free=False, bounds=b, unit="mm", ui={"group": s_group, "order": n_i}
        )
        for n_i, (s_name, d_value, b, s_group, _) in enumerate(PARAMS)
    }
    return Part(
        name="backpack_table_hook",
        params=params,
        materials=[MaterialRegion(material_id=1, name="pla_3dprint", sdf_tree=build_tree())],
        metadata={
            "units": "mm",
            "description": "Clamp-on C hook: tapered top bar on the table, spine down the edge, "
            "jaw under the table looping back into a front-loading handle hook",
            "frame": "profile in XY as printed; +X away from table, +Y up installed, "
            "z = width on bed",
            "process": "FDM, Prusa Core One, Prusament PLA, printed on its side",
            "param_meanings": {s_name: s_doc for s_name, *_, s_doc in PARAMS},
        },
    )


def main():
    logging.basicConfig(level=logging.INFO)
    part = build_part()
    tree = part.materials[0].sdf_tree
    sdm_path = OUT / "backpack_table_hook.sdm"
    save(part, sdm_path)
    # sdm-core 1.0 refuses a legacy `couplings` key, even
    # empty; this part declares none, so drop it and both versions load it.
    doc = json.loads(sdm_path.read_text())
    assert doc.pop("couplings", []) == []
    sdm_path.write_text(json.dumps(doc, indent=2) + "\n")
    load(sdm_path)

    d = DEFAULTS
    assert d["d_lip_height"] > d["d_fillet_hook"] + d["d_wall"] / 2, "lip too short for its bend"
    assert d["d_table_thickness"] + d["d_table_clearance"] > 2 * d["d_fillet_table"], (
        "table bends overlap"
    )
    field = bind_sdf(tree, part)
    d_tc = d["d_table_thickness"] + d["d_table_clearance"]
    z_mid = d["d_width"] / 2
    loop_cy = -d_tc - d["d_wall"] - d["d_throat"] / 2
    leg_top = loop_cy - d["d_throat"] / 2
    # The table must fit with a knife-sharp edge: sample the slab 0.01 mm inside
    # its faces (the top bar rests on the table top, so that face is contact).
    xs, ys = np.meshgrid(
        np.linspace(-200, -0.01, 500), np.linspace(-d["d_table_thickness"] + 0.01, -0.01, 100)
    )
    table = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, z_mid)], -1).astype(np.float32)
    d_table_min = float(np.min(np.asarray(field(table))))
    assert d_table_min > 0, f"part intrudes into the table by {-d_table_min:.3f} mm"
    # The real desk has ~3.25 mm rounds on its edges. Check the bends clear that
    # rounded slab too, and report the gap they leave, so the inside bends never
    # blend into the desk's own radius.
    d_desk_r = D_DESK_EDGE_RADIUS_MM
    d_t = d["d_table_thickness"]
    b_keep = ~((xs > -d_desk_r) & ((ys > -d_desk_r) | (ys < -d_t + d_desk_r)))
    ang = np.linspace(0, math.pi / 2, 200)
    arc_top = np.stack(
        [-d_desk_r + (d_desk_r - 0.01) * np.cos(ang), -d_desk_r + (d_desk_r - 0.01) * np.sin(ang)],
        -1,
    )
    arc_bot = arc_top * [1, -1] + [0, -d_t]
    desk_xy = np.concatenate([np.stack([xs[b_keep], ys[b_keep]], -1), arc_top, arc_bot])
    desk = np.column_stack([desk_xy, np.full(len(desk_xy), z_mid)]).astype(np.float32)
    d_desk_min = float(np.min(np.asarray(field(desk))))
    assert d_desk_min > 0, f"part intrudes into the rounded desk by {-d_desk_min:.3f} mm"
    # Gap at the middle of each desk round (its ends touch the flat faces).
    d_desk_corner_gap = float(np.min(np.asarray(field(desk[[-300, -100]]))))
    # Field fairness: a seam between pieces shows up as |grad| != 1 inside.
    # Check the band the side-edge rounding reads (depth up to ~r). Stay clear
    # of the top-bar tip: its square corners and thin section put real medial-
    # axis kinks inside that band, and those are not seams.
    rng = np.random.default_rng(0)
    cand = np.stack(
        [rng.uniform(-80, 20, 400000), rng.uniform(-80, 20, 400000), np.full(400000, z_mid)], -1
    )
    dv = np.asarray(field(cand.astype(np.float32)))
    b_band = (dv < -0.2) & (dv > -1.3) & (cand[:, 0] > -d["d_top_length"] + 2)
    band = cand[b_band][:20000]
    d_h = 1e-2
    grad = np.stack(
        [
            (
                np.asarray(field((band + d_h * np.eye(3)[k]).astype(np.float32)))
                - np.asarray(field((band - d_h * np.eye(3)[k]).astype(np.float32)))
            )
            / (2 * d_h)
            for k in range(2)
        ],
        -1,
    )
    gnorm = np.linalg.norm(grad, axis=-1)
    d_grad_dev = float(np.max(np.abs(gnorm - 1)))
    assert d_grad_dev < 0.05, (
        f"field not exact near the surface: max | |grad|-1 | = {d_grad_dev:.3f}"
    )
    probes = {
        "top_bar_mid": ([-30, 2.0, z_mid], True),
        "spine": (
            [d["d_table_clearance"] + d["d_fillet_table"] + d["d_wall"] / 2, -12, z_mid],
            True,
        ),
        "jaw": ([-15, -d_tc - 5, z_mid], True),
        "loop_back": (
            [-d["d_jaw_depth"] - d["d_throat"] / 2 - d["d_wall"] / 2, loop_cy, z_mid],
            True,
        ),
        "lower_leg": ([-15, leg_top - d["d_wall"] / 2, z_mid], True),
        "lip": (
            [-d["d_lip_setback"] - d["d_wall"] / 2, leg_top + d["d_lip_height"] / 2, z_mid],
            True,
        ),
        "throat_center": ([-15, loop_cy, z_mid], False),
        "handle_entry": ([-5, loop_cy + d["d_throat"] / 2 - 3, z_mid], False),
        "above_top_bar_tip": ([-65, 5.0, z_mid], False),
    }
    pts = np.array([p for p, _ in probes.values()], dtype=np.float32)
    got = np.asarray(field(pts)) < 0
    want = np.array([b for _, b in probes.values()])
    bad = [k for k, g, w in zip(probes, got, want, strict=True) if g != w]
    assert not bad, f"probe mismatch: {bad}"
    print("SDM geometry verified. Exporting STL...", flush=True)

    # sdm-core's export grid starts exactly 2 voxels below the part's bbox, so
    # a flat face lying on the bbox (the bed face, the top face, the tip) sits
    # on a grid plane and marching cubes leaves zero-area slivers there. Export
    # a copy whose sampling box is grown by a fraction of a voxel instead; the
    # saved .sdm is untouched.
    box = resolve_bbox(tree, part)
    d_shift = 0.37 * D_VOXEL_MM
    export_copy = copy.deepcopy(part)
    export_copy.metadata["bbox"] = [
        (box.min_pt - d_shift).tolist(),
        (box.max_pt + d_shift).tolist(),
    ]
    paths = export_part(
        export_copy,
        OUT,
        voxel_size=D_VOXEL_MM,
        fmt="stl",
        decimate=DecimateConfig(simplify_error_mm=0.02),
        stamp="date",
    )
    mesh = trimesh.load_mesh(paths[0], process=True)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.is_volume
    assert mesh.body_count == 1
    d_exp_w = d["d_width"]
    assert abs(mesh.bounds[0][2]) < 0.05 and abs(mesh.bounds[1][2] - d_exp_w) < 0.05, mesh.bounds
    d_mass_g = mesh.volume / 1000 * 1.24  # Prusament PLA 1.24 g/cm^3, solid
    report = {
        "sdm": sdm_path.name,
        "stl": paths[0].name,
        "units": "mm",
        "params": DEFAULTS,
        "watertight": bool(mesh.is_watertight),
        "bodies": int(mesh.body_count),
        "triangles": len(mesh.faces),
        "bounds_mm": np.round(mesh.bounds, 3).tolist(),
        "volume_mm3": round(float(mesh.volume), 1),
        "solid_mass_g": round(d_mass_g, 1),
        "table_clearance_min_mm": round(d_table_min, 3),
        "table_edge_assumed": "knife-sharp (worst case)",
        "desk_edge_radius_mm": D_DESK_EDGE_RADIUS_MM,
        "desk_rounded_clearance_min_mm": round(d_desk_min, 3),
        "desk_corner_gap_mm": round(d_desk_corner_gap, 3),
        "field_max_grad_deviation": round(d_grad_dev, 4),
        "probes_ok": list(probes),
        "strength": bending_report(d),
        "export_voxel_mm": D_VOXEL_MM,
    }
    (OUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
