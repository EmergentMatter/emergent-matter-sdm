"""3-D TET10 FEA on an extruded mesh: the handle-width and layer-adhesion check.

A TetGen mesh of the exported STL surface was tried first; keeping the decimated
surface triangulation left sliver tets (1 % below quality 0.004) that put
spurious spikes at the side edges. The hook is an extrusion, so here the 3-D
mesh is the 2-D profile mesh swept across the width: prisms split into three
tets by the minimum-global-index rule (conforming on every shared face), and
z-planes placed on the 1 mm skin boundaries so the skins are exact. The 1 mm
side-edge rounding is left out (no load reaches those edges).

Output is the ratio of 3-D to 2-D peak stress at the same element size, for the
solid part and for 6 perimeters / 40 % gyroid: the knockdown for the handle
being narrower than the hook, applied to the converged 2-D results.

Contact: penalty table (stiffness 1e3 x the largest diagonal) on the underside
nodes, semismooth Newton on the active set (nodes pressing into the table).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import importlib
import json
import time

import fea_common as fc
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from jax_fem.generate_mesh import Mesh
from meshing import tet4_to_tet10, tri6_mesh_polygon

f2d = importlib.import_module("01_fea_2d")

D_H_FINE, D_H_COARSE = 2.0, 3.0  # matched by the FEA_H_FINE=2.0 2-D run for the 3-D/2-D ratio
D_HANDLE_W = 30.0
D_SKIN = 1.0


def extrude(p2, t3, zs):
    """TRI3 mesh (p2, t3) swept through z-planes zs -> conforming TET4 mesh."""
    n2 = len(p2)
    pts = np.vstack([np.c_[p2, np.full(n2, z)] for z in zs])
    t = np.sort(t3, axis=1)  # v0 < v1 < v2 by global id -> consistent diagonals
    tets = []
    for k in range(len(zs) - 1):
        b, u = t + k * n2, t + (k + 1) * n2
        tets += [
            np.c_[b[:, 0], b[:, 1], b[:, 2], u[:, 0]],
            np.c_[b[:, 1], b[:, 2], u[:, 0], u[:, 1]],
            np.c_[b[:, 2], u[:, 0], u[:, 1], u[:, 2]],
        ]
    tets = np.vstack(tets)
    a, bb, c, d = (pts[tets[:, i]] for i in range(4))
    vol = np.einsum("ij,ij->i", np.cross(bb - a, c - a), d - a)
    tets[vol < 0] = tets[vol < 0][:, [0, 2, 1, 3]]
    return pts, tets


def run(pts, cells, e_q, g, d, problem, warm_x=None):
    n = 3 * len(pts)
    k, _ = fc.assemble(problem, n)
    b_under = (np.abs(pts[:, 1]) < 0.02) & (
        pts[:, 0] < g["c"] + 0.02
    )  # outline y is contoured to ~1e-3
    d_w = d["d_width"]
    # Friction holds at the tip-most underside nodes: x at two, z at one.
    d_x_tip = pts[b_under, 0].min()
    cand = np.where(b_under & (pts[:, 0] < d_x_tip + 0.5))[0]
    i_f1 = cand[np.argmin(np.abs(pts[cand, 2] - d_w * 0.2))]
    i_f2 = cand[np.argmin(np.abs(pts[cand, 2] - d_w * 0.8))]
    fix = np.array([3 * i_f1, 3 * i_f2, 3 * i_f1 + 2])
    # Handle load: upward-facing leg-top faces with centroid in the patch.
    x0, x1 = g["cx"], g["cx"] + f2d.D_PATCH
    z0, z1 = d_w / 2 - D_HANDLE_W / 2, d_w / 2 + D_HANDLE_W / 2
    bdy = fc.boundary_entities(pts, cells)
    tri = pts[bdy[:, :3]]
    cen = tri.mean(axis=1)
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    b_face = (
        (np.abs(cen[:, 1] - g["y_leg_top"]) < 0.05)
        & (np.abs(nrm[:, 1]) > 0.95)
        & (cen[:, 0] > x0)
        & (cen[:, 0] < x1)
        & (cen[:, 2] > z0)
        & (cen[:, 2] < z1)
    )
    d_force = 15 * fc.D_G
    f, d_area = fc.face_load_3d(pts, bdy[b_face], [0.0, -1.0, 0.0])
    f *= d_force / d_area
    free = np.setdiff1d(np.arange(n), fix)
    dofs_c = 3 * np.where(b_under)[0] + 1
    d_pen = 1e3 * float(k.diagonal().max())
    # Start from the matching 2-D contact band when known: each iteration is a
    # full factorisation (~90 s), and from "everything touches" it takes ~20.
    x_c0 = pts[dofs_c // 3, 0]
    b_act = (
        np.ones(len(dofs_c), dtype=bool)
        if warm_x is None
        else (x_c0 >= warm_x[0] - 1) & (x_c0 <= warm_x[1] + 1)
    )
    for n_it in range(40):
        pen = np.zeros(n)
        pen[dofs_c[b_act]] = d_pen
        a = (k + sp.diags(pen)).tocsc()[free][:, free]
        u = np.zeros(n)
        u[free] = spla.splu(a, permc_spec="COLAMD").solve(f[free])
        b_new = u[dofs_c] < 0.0  # pressing into the table
        print(
            f"    contact iter {n_it}: active {int(b_act.sum())} -> {int(b_new.sum())}", flush=True
        )
        if (b_new == b_act).all():
            break
        b_act = b_new
    lam = -d_pen * np.minimum(u[dofs_c], 0.0) * b_act
    d_fy = float(lam.sum())
    x_c = pts[dofs_c // 3, 0]
    fy = f.reshape(-1, 3)[:, 1]
    d_x_load = float((fy * pts[:, 0]).sum() / fy.sum())
    d_x_sup = float((lam * x_c).sum() / lam.sum())
    assert abs(d_fy / d_force - 1) < 1e-3, (d_fy, d_force)
    assert abs(d_x_sup - d_x_load) < 0.1, (d_x_sup, d_x_load)
    d_pen_depth = float(-u[dofs_c].min())
    sol = u.reshape(-1, 3)
    info = {
        "contact_iterations": n_it,
        "active_nodes": int(b_act.sum()),
        "support_total_n": round(d_fy, 3),
        "x_load_mm": round(d_x_load, 3),
        "x_support_mm": round(d_x_sup, 3),
        "max_penetration_mm": d_pen_depth,
        "patch_area_mm2": round(d_area, 2),
        "contact_x_range_mm": [
            round(float(x_c[b_act].min()), 2),
            round(float(x_c[b_act].max()), 2),
        ],
    }
    return sol, info


def summarise(problem, sol, cells, g, e_cell_solid):
    sig = fc.cell_stress(problem, sol, e_cell_solid)
    s1 = fc.max_principal(sig)
    q = np.asarray(problem.physical_quad_points)
    masks = f2d.regions(q[..., :2], g)
    reg = {
        k: {
            "max_principal_mpa": round(float(s1[m].max()), 2),
            "p99_9_mpa": round(float(np.percentile(s1[m], 99.9)), 2),
        }
        for k, m in masks.items()
    }
    return reg, sig, q


def main():
    t0 = time.time()
    part = fc.load_part()
    loops, field, d = f2d.profile_loops(part, max(0.25, D_H_FINE / 2))  # same as the 2-D reference
    g = f2d.geometry_refs(d)
    p2, t3 = tri6_mesh_polygon(loops, D_H_FINE, D_H_COARSE, 3.0, order=1)
    d_w = d["d_width"]
    zs = np.unique(np.r_[0.0, D_SKIN, np.linspace(D_SKIN, d_w - D_SKIN, 10), d_w - D_SKIN, d_w])
    p4, c4 = extrude(p2, t3, zs)
    pts, cells = tet4_to_tet10(p4, c4)
    print(
        f"[3dx] 2-D {len(t3)} tri -> {len(c4)} tets, {len(pts)} TET10 nodes, {3 * len(pts)} dofs",
        flush=True,
    )
    mesh = Mesh(pts, cells, ele_type="TET10")
    problem = fc.Elasticity(
        mesh=mesh,
        vec=3,
        dim=3,
        ele_type="TET10",
        location_fns=[lambda p: p[0] > 1e9],
        additional_info=([np.zeros(3)],),
    )
    q = np.asarray(problem.physical_quad_points)
    q3 = q.reshape(-1, 3).copy()
    q3[:, 2] = d_w / 2  # in-plane depth from the profile SDF at mid-width
    depth = -np.asarray(field(q3.astype(np.float32))).reshape(q.shape[:2])
    zq = q[..., 2]
    out = {
        "mesh": {
            "tri2d": int(len(t3)),
            "tets": int(len(c4)),
            "nodes": int(len(pts)),
            "z_planes_mm": np.round(zs, 3).tolist(),
            "h_fine_mm": D_H_FINE,
        },
        "configs": {},
    }
    e_solid = np.full(len(cells), fc.D_E_PLA)
    res_2d = json.loads((fc.FEA / "results_2d_h2.json").read_text())["configs"]
    s_cfgs = os.environ.get("FEA_3D_CONFIGS", "solid_100,6perim_40gyroid").split(",")
    p_out = fc.FEA / "results_3d_extruded.json"
    if p_out.exists():  # keep configs finished by an earlier run
        out["configs"] = json.loads(p_out.read_text()).get("configs", {})
    for s_cfg in s_cfgs:
        cfg = f2d.CONFIGS[s_cfg]
        if cfg["n_perim"] is None:
            e_q = np.full(depth.shape, fc.D_E_PLA)
        else:
            d_t_wall = f2d.wall_thickness(cfg["n_perim"])
            d_ec = f2d.infill_modulus_ratio(cfg["d_rho"])
            b_solid = (depth < d_t_wall) | (zq < cfg["d_skin"]) | (zq > d_w - cfg["d_skin"])
            e_q = np.where(b_solid, fc.D_E_PLA, fc.D_E_PLA * d_ec)
        problem.internal_vars = (jnp.asarray(e_q),)
        x_band = res_2d[s_cfg]["cases"]["A_innermost"]["contact"]["contact_x_mm"]
        sol, info = run(pts, cells, e_q, g, d, problem, warm_x=(min(x_band), max(x_band)))
        reg, sig, _ = summarise(problem, sol, cells, g, e_solid)
        # Stress in the layer each point actually is: solid, or infill (scaled).
        if cfg["n_perim"] is not None:
            b_core = ~b_solid
            s_core = d_ec * np.abs(np.linalg.eigvalsh(sig)).max(-1)
            s1_solid = np.where(b_solid, fc.max_principal(sig), 0.0)
            reg_solid = float(s1_solid.max())
            core = {
                "infill_max_abs_principal_mpa": round(float(s_core[b_core].max()), 3),
                "infill_utilisation": round(
                    float(s_core[b_core].max()) / f2d.infill_strength(cfg["d_rho"]), 3
                ),
                "solid_material_peak_s1_mpa": round(reg_solid, 2),
                "solid_material_p99_9_s1_mpa": round(
                    float(np.percentile(s1_solid[b_solid], 99.9)), 2
                ),
            }
        else:
            core = {}
        b_mat = np.ones(sig.shape[:2], dtype=bool) if cfg["n_perim"] is None else b_solid
        s_zz = np.where(b_mat, sig[..., 2, 2], 0.0)
        tau = np.where(b_mat, np.hypot(sig[..., 0, 2], sig[..., 1, 2]), 0.0)
        out["configs"][s_cfg] = {
            "contact": info,
            "regions": reg,
            **core,
            "interlayer": {
                "max_sigma_zz_tension_mpa": round(float(s_zz.max()), 3),
                "max_interlayer_shear_mpa": round(float(tau.max()), 3),
            },
            "max_displacement_mm": round(float(np.linalg.norm(sol, axis=1).max()), 3),
        }
        print(
            f"[3dx] {s_cfg}: {json.dumps(out['configs'][s_cfg]['regions']['anywhere'])} "
            f"{json.dumps(core)} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
        p_out.write_text(json.dumps(out, indent=2) + "\n")  # after every config
        # Full field for the renderer (04_render_3d.py).
        b_mat_save = np.ones(sig.shape[:2], dtype=bool) if cfg["n_perim"] is None else b_solid
        np.savez_compressed(
            fc.FEA / f"_fea3dx_{s_cfg}.npz",
            pts=pts,
            cells=cells,
            sol=sol,
            s1_cell=np.where(b_mat_save, fc.max_principal(sig), 0.0).max(axis=1),
            s_zz_cell=np.where(b_mat_save, sig[..., 2, 2], 0.0).max(axis=1),
        )
    print(f"[3dx] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
