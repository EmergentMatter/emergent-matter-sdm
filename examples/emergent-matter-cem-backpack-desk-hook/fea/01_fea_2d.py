"""2-D plane-stress FEA of the hook profile, solid and with slicer infill.

Why 2-D is exact enough here: the hook is an extrusion loaded in its own plane,
and it prints lying on its side, so every layer is a copy of the profile. The
slicer turns each layer into solid perimeter walls that follow the outline and
sparse infill inside, and the first/last few layers (the two side faces) are
fully solid skins. Through the width, at any (x, y) inside the walls, the part
is a sandwich: skin | infill | skin. In-plane strain is uniform through the
width, so the effective in-plane modulus there is the thickness-weighted mix,
and the stress in each layer is its own modulus times that strain.

Contact: the clip is not bolted, it rests on the table. The top-bar underside
is a unilateral support: solved with every underside node held vertically,
then nodes the table would have to pull down on are released, and repeat until
every held node is in compression. One node is held horizontally (friction).

Load: the handle as a 6 mm patch on the leg's flat, at the two ends of where it
can rest under gravity. Case A is the innermost spot (longest lever onto the
table bends), case B is against the lip.
"""

from __future__ import annotations

import os

# Before anything imports jax: double precision for the direct solves.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import json
import time

import contourpy
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import fea_common as fc
import matplotlib.pyplot as plt
import numpy as np
from jax_fem.generate_mesh import Mesh
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation
from meshing import tri6_mesh_polygon
from software_defined_matter.grid_sampling import bind_sdf

# mm, element size within D_BAND of the outline (resolves the walls). Set
# FEA_H_FINE=0.5 for the mesh-convergence run (writes results_2d_h0.5.json).
D_H_FINE = float(os.environ.get("FEA_H_FINE", "0.25"))
D_H_COARSE = max(0.8, 1.5 * D_H_FINE)  # 3.0 at FEA_H_FINE=2.0, matching 02_fea_3d
D_BAND = 3.0
D_PATCH = 6.0  # mm, handle contact length along the leg

# Slicer configurations. Wall thickness for n perimeters at PrusaSlicer's
# 0.45 mm width / 0.2 mm layer: 0.45 + (n - 1) * 0.407 mm. Skins are the
# solid top/bottom layers, which lying on its side are the two side faces.
CONFIGS = {
    "solid_100": {"n_perim": None, "d_skin": None, "d_rho": 1.0},
    "6perim_40gyroid": {"n_perim": 6, "d_skin": 1.0, "d_rho": 0.40},
    "4perim_25gyroid": {"n_perim": 4, "d_skin": 1.0, "d_rho": 0.25},
    "3perim_15gyroid": {"n_perim": 3, "d_skin": 0.8, "d_rho": 0.15},
    "2perim_15gyroid": {"n_perim": 2, "d_skin": 0.8, "d_rho": 0.15},
    # More walls, less infill: the walls carry the bending, the infill the shear.
    "6perim_20gyroid": {"n_perim": 6, "d_skin": 1.0, "d_rho": 0.20},
    "8perim_20gyroid": {"n_perim": 8, "d_skin": 1.0, "d_rho": 0.20},
    "8perim_40gyroid": {"n_perim": 8, "d_skin": 1.0, "d_rho": 0.40},
    # Sensitivity: stiffer infill law (E*/Es = rho^1.5, the other end of the range).
    "6perim_40gyroid_exp1.5": {"n_perim": 6, "d_skin": 1.0, "d_rho": 0.40, "d_exp": 1.5},
    "6perim_20gyroid_exp1.5": {"n_perim": 6, "d_skin": 1.0, "d_rho": 0.20, "d_exp": 1.5},
    "4perim_25gyroid_exp1.5": {"n_perim": 4, "d_skin": 1.0, "d_rho": 0.25, "d_exp": 1.5},
    "2perim_15gyroid_exp1.5": {"n_perim": 2, "d_skin": 0.8, "d_rho": 0.15, "d_exp": 1.5},
}


def wall_thickness(n_perim):
    return 0.45 + (n_perim - 1) * 0.407


def infill_modulus_ratio(d_rho, d_exp=2.0):
    # Gibson-Ashby, bending-dominated cellular solid: E*/Es = rho^2. Reported
    # FDM gyroid exponents run 1.5 to 2; the low end of stiffness is the
    # conservative choice for the walls, which then carry more of the load.
    return d_rho**d_exp


def infill_strength(d_rho):
    # Gibson-Ashby open-cell: sigma*/sigma_s = 0.3 rho^1.5 (conservative for gyroid).
    return 0.3 * fc.D_YIELD_PLA * d_rho**1.5


def profile_loops(part, d_spacing=0.25):
    """Outline of the profile at mid-width, from the part's own SDF."""
    d = {s: p.value for s, p in part.params.items()}
    field = bind_sdf(part.materials[0].sdf_tree, part)
    d_step = 0.05
    xs = np.arange(-75.0, 22.0, d_step)
    ys = np.arange(-78.0, 16.0, d_step)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, d["d_width"] / 2)], -1).astype(
        np.float32
    )
    vals = np.concatenate(
        [np.asarray(field(pts[i : i + 400000])) for i in range(0, len(pts), 400000)]
    )
    lines = contourpy.contour_generator(xs, ys, vals.reshape(gx.shape)).lines(0.0)
    loops = []
    for line in lines:
        loop = line[:-1] if np.allclose(line[0], line[-1]) else line
        # Resample to d_spacing: at 0.25 mm the chord error on the 3 mm lip bend
        # is < 0.003 mm; coarse reference meshes use h/2.
        seg = np.linalg.norm(np.diff(np.vstack([loop, loop[:1]]), axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        n = int(s[-1] / d_spacing)
        t = np.linspace(0, s[-1], n, endpoint=False)
        closed = np.vstack([loop, loop[:1]])
        loops.append(np.stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])], -1))
    loops.sort(key=lambda lp: -len(lp))
    assert len(loops) == 1, f"expected one outline, got {len(loops)}"
    return loops, field, d


def geometry_refs(d):
    c = d["d_table_clearance"]
    rf, wall, gap = d["d_fillet_table"], d["d_wall"], d["d_throat"]
    d_tc = d["d_table_thickness"] + c
    cx, cy = -d["d_jaw_depth"], -d_tc - wall - gap / 2
    y_leg_top = cy - gap / 2
    x_lip_in = -d["d_lip_setback"] - wall
    return {
        "c": c,
        "rf": rf,
        "wall": wall,
        "d_tc": d_tc,
        "cx": cx,
        "cy": cy,
        "y_leg_top": y_leg_top,
        "x_lip_in": x_lip_in,
        "x_flat_end": x_lip_in - d["d_fillet_hook"],
        "c1": (c, -rf),
        "c2": (c, -d_tc + rf),
        "r_loop_out": gap / 2 + wall,
    }


def regions(q, g):
    """Named masks over (..., 2) points: where each result is reported."""
    x, y = q[..., 0], q[..., 1]
    r1 = np.hypot(x - g["c1"][0], y - g["c1"][1])
    r2 = np.hypot(x - g["c2"][0], y - g["c2"][1])
    rl = np.hypot(x - g["cx"], y - g["cy"])
    top = (x >= g["c1"][0] - 0.5) & (y >= g["c1"][1] - 0.5) & (r1 < g["rf"] + g["wall"] + 0.5)
    bot = (x >= g["c2"][0] - 0.5) & (y <= g["c2"][1] + 0.5) & (r2 < g["rf"] + g["wall"] + 0.5)
    loop = (x <= g["cx"] + 0.5) & (rl < g["r_loop_out"] + 0.5)
    top_bar = (y > -0.5) & (x < g["c1"][0] - 0.5)
    return {
        "top_bend": top,
        "bottom_bend": bot,
        "loop": loop,
        "top_bar": top_bar,
        "anywhere": np.ones_like(x, dtype=bool),
    }


def held_nodes(b_mask):
    """Location fn (point, index) -> held; jax-fem counts arguments, so no defaults."""
    mask = jnp.asarray(b_mask)

    def loc(p, i):
        return mask[i]

    return loc


def contact_setup(points, g):
    """Table-contact dofs (underside, vertical) and the friction hold (tip node, x)."""
    n_dim = points.shape[1]
    b_under = (np.abs(points[:, 1]) < 0.02) & (points[:, 0] < g["c"] + 0.02)
    i_under = np.where(b_under)[0]
    return b_under, i_under * n_dim + 1


def solve_config(points, cells, e_q, g, d, cases, d_force):
    """Assemble once, condense onto the table contact, solve every load case."""
    mesh = Mesh(points, cells, ele_type="TRI6")
    problem = fc.Elasticity(
        mesh=mesh,
        vec=2,
        dim=2,
        ele_type="TRI6",
        location_fns=[lambda p: p[0] > 1e9],
        additional_info=([np.zeros(2)],),
    )
    problem.internal_vars = (jnp.asarray(e_q),)
    k, _ = fc.assemble(problem, 2 * len(points))
    b_under, dofs_c = contact_setup(points, g)
    i_fric = int(np.argmin(np.where(b_under, points[:, 0], np.inf)))  # tip-most node, x
    contact = fc.UnilateralContact(k, dofs_c, [2 * i_fric])
    d_b = d["d_width"]
    out = {}
    for s_case, (x0, x1) in cases.items():
        b_patch = (
            (np.abs(points[:, 1] - g["y_leg_top"]) < 0.02)
            & (points[:, 0] > x0 - 1e-6)
            & (points[:, 0] < x1 + 1e-6)
        )
        f, d_len = fc.patch_load(points, cells, b_patch, [0.0, -1.0])
        f *= d_force / (
            d_b * d_len
        )  # exactly the load on the edges the patch selected (per unit width)
        u, lam, u_c, info = contact.solve(f)
        d_fy = float(lam.sum() * d_b)
        assert abs(d_fy / d_force - 1) < 1e-3, f"support total {d_fy} N vs load {d_force} N"
        # Moment balance about the origin: the support resultant lines up under the load.
        fy = f.reshape(-1, 2)[:, 1]
        x_c = points[np.where(b_under)[0], 0]
        d_x_load = float((fy * points[:, 0]).sum() / fy.sum())
        d_x_support = float((lam * x_c).sum() / lam.sum())
        assert abs(d_x_support - d_x_load) < 0.05, (d_x_support, d_x_load)
        assert info["min_support_force_rel"] > -1e-4 and info["force_at_lifted_rel"] < 1e-3, info
        held = np.where(b_under)[0][u_c <= 1e-9]
        info.update(
            patch_length_mm=round(d_len, 3),
            support_total_n=round(d_fy, 3),
            x_load_mm=round(d_x_load, 3),
            x_support_resultant_mm=round(d_x_support, 3),
            contact_x_mm=np.round(np.unique(np.round(points[held, 0], 1)), 1).tolist(),
        )
        out[s_case] = (u.reshape(-1, 2), info)
    return problem, out


def main():
    t0 = time.time()
    part = fc.load_part()
    loops, field, d = profile_loops(part, max(0.25, D_H_FINE / 2))
    g = geometry_refs(d)
    pts, cells = tri6_mesh_polygon(loops, D_H_FINE, D_H_COARSE, D_BAND)
    print(
        f"[2d] mesh: {len(pts)} nodes, {len(cells)} TRI6, outline {len(loops[0])} pts", flush=True
    )

    # Quad-point geometry for the layered material.
    mesh = Mesh(pts, cells, ele_type="TRI6")
    probe = fc.Elasticity(
        mesh=mesh,
        vec=2,
        dim=2,
        ele_type="TRI6",
        location_fns=[lambda p: p[0] > 1e9],
        additional_info=([np.zeros(2)],),
    )
    q = np.asarray(probe.physical_quad_points)  # (cells, quads, 2)
    qf = q.reshape(-1, 2)
    q3 = np.concatenate([qf, np.full((len(qf), 1), d["d_width"] / 2)], 1).astype(np.float32)
    depth = -np.asarray(field(q3)).reshape(q.shape[:2])  # in-plane distance to the outline
    masks = regions(q, g)

    d_w15 = 15 * fc.D_G
    cases = {
        "A_innermost": (g["cx"], g["cx"] + D_PATCH),
        "B_at_lip": (g["x_flat_end"] - D_PATCH, g["x_flat_end"]),
    }
    results = {
        "load_n_for_15kg": d_w15,
        "mesh": {
            "nodes": int(len(pts)),
            "cells": int(len(cells)),
            "h_fine_mm": D_H_FINE,
            "h_coarse_mm": D_H_COARSE,
        },
        "configs": {},
    }
    saved = {}
    for s_cfg, cfg in CONFIGS.items():
        if cfg["n_perim"] is None:
            e_q = np.full(depth.shape, fc.D_E_PLA)
            d_t_wall, d_ec = None, 1.0
        else:
            d_t_wall = wall_thickness(cfg["n_perim"])
            d_ec = infill_modulus_ratio(cfg["d_rho"], cfg.get("d_exp", 2.0))
            d_frac_skin = 2 * cfg["d_skin"] / d["d_width"]
            d_e_core_zone = fc.D_E_PLA * (d_frac_skin + (1 - d_frac_skin) * d_ec)
            e_q = np.where(depth < d_t_wall, fc.D_E_PLA, d_e_core_zone)
        b_core = np.zeros(depth.shape, dtype=bool) if d_t_wall is None else depth >= d_t_wall
        res_cfg = {
            "wall_thickness_mm": d_t_wall,
            "skin_mm": cfg["d_skin"],
            "infill_rho": cfg["d_rho"],
            "infill_E_ratio": d_ec,
            "infill_E_exponent": cfg.get("d_exp", 2.0),
            "cases": {},
        }
        problem, solved = solve_config(pts, cells, e_q, g, d, cases, d_w15)
        for s_case, (sol, info) in solved.items():
            # Stress in solid PLA at every point (walls, and the skins over the core).
            sig_s = fc.cell_stress(problem, sol, np.full(len(cells), fc.D_E_PLA))
            s1 = fc.max_principal(sig_s)
            vm = fc.von_mises(sig_s)
            reg = {
                k: {
                    "max_principal_mpa": round(float(s1[m].max()), 2),
                    "von_mises_mpa": round(float(vm[m].max()), 2),
                }
                for k, m in masks.items()
            }
            core = {}
            if b_core.any():
                s_core = d_ec * np.abs(np.linalg.eigvalsh(sig_s)).max(-1)
                d_core_max = float(s_core[b_core].max())
                core = {
                    "infill_max_abs_principal_mpa": round(d_core_max, 3),
                    "infill_strength_mpa": round(infill_strength(cfg["d_rho"]), 3),
                    "infill_utilisation": round(d_core_max / infill_strength(cfg["d_rho"]), 3),
                }
            u = np.linalg.norm(sol, axis=1)
            b_jaw_top = (
                (np.abs(pts[:, 1] + g["d_tc"]) < 0.02)
                & (pts[:, 0] < g["c"])
                & (pts[:, 0] > g["cx"])
            )
            b_spine_in = np.abs(pts[:, 0] - (g["c"] + g["rf"])) < 0.02
            res_cfg["cases"][s_case] = {
                "patch_x_mm": [round(v, 2) for v in cases[s_case]],
                "contact": info,
                "regions": reg,
                **core,
                "max_displacement_mm": round(float(u.max()), 3),
                "jaw_rise_toward_table_mm": round(float(sol[b_jaw_top, 1].max()), 4),
                "table_gap_under_jaw_mm": d["d_table_clearance"],
                "spine_move_toward_edge_mm": round(float(-sol[b_spine_in, 0].min()), 4),
                "edge_gap_mm": g["c"] + g["rf"],
            }
            saved[(s_cfg, s_case)] = (sol, s1)
            print(
                f"[2d] {s_cfg:17s} {s_case:12s} walls/skins peak s1="
                f"{reg['anywhere']['max_principal_mpa']:6.2f} MPa  "
                f"{('infill util=' + str(core['infill_utilisation'])) if core else ''}  "
                f"contact x={info['contact_x_mm'][:3]}..{info['contact_x_mm'][-3:]}  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
        results["configs"][s_cfg] = res_cfg

    s_suffix = "" if D_H_FINE == 0.25 else f"_h{D_H_FINE:g}"
    (fc.FEA / f"results_2d{s_suffix}.json").write_text(json.dumps(results, indent=2) + "\n")
    if not s_suffix:
        plot(pts, cells, saved, g)
    print(f"[2d] done in {time.time() - t0:.0f}s")


def plot(pts, cells, saved, g):
    tri = Triangulation(pts[:, 0], pts[:, 1], cells[:, :3])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    keys = [("solid_100", "A_innermost"), ("6perim_40gyroid", "A_innermost")]
    for ax, key in zip(axes, keys, strict=True):
        sol, s1 = saved[key]
        d_scale = 10.0
        tri_d = Triangulation(
            pts[:, 0] + d_scale * sol[:, 0], pts[:, 1] + d_scale * sol[:, 1], cells[:, :3]
        )
        tpc = ax.tripcolor(tri_d, s1.max(axis=1), cmap="inferno", vmin=0, vmax=15, shading="flat")
        ax.triplot(tri, color="#999", lw=0.05, alpha=0.2)
        d_tt = g["d_tc"] - g["c"]
        ax.add_patch(Rectangle((-80, -d_tt), 80, d_tt, color="#d9c4a0", alpha=0.35, zorder=0))
        ax.set_aspect("equal")
        ax.set_title(
            f"{key[0]}, 15 kg, handle innermost\n"
            f"max principal stress in solid PLA (deformed x{d_scale:g})"
        )
        fig.colorbar(tpc, ax=ax, label="MPa", shrink=0.8)
    fig.tight_layout()
    fig.savefig(fc.FEA / "stress_2d.png", dpi=140)


if __name__ == "__main__":
    main()
