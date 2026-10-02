"""Validate the FEA path on a cantilever with a known answer before using it.

Checks, in 2-D plane stress (TRI6, gmsh) and 3-D (TetGen TET4 -> our TET10):
  * the TET10 node order we build matches what jax-fem expects,
  * the sign convention of tractions and of the reaction recovery,
  * bending accuracy: tip deflection vs Timoshenko beam theory, and the
    fibre stress at mid-span vs Mc/I.
A wrong node order or sign shows up here as an error of order 1, not a few %.
"""

from __future__ import annotations

import os

# Before anything imports jax: double precision for the direct solves.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import json

import fea_common as fc
import jax.numpy as jnp
import numpy as np
import trimesh
from jax_fem.generate_mesh import Mesh
from jax_fem.solver import solver
from meshing import tet4_to_tet10, tet_mesh_surface, tri6_mesh_polygon

D_L, D_H, D_B = 100.0, 12.0, 10.0  # length, depth (y), width (z)
D_P = 50.0  # N, tip shear load, -y


def beam_theory():
    d_e = fc.D_E_PLA
    d_g = d_e / (2 * (1 + fc.D_NU_PLA))
    d_i = D_B * D_H**3 / 12
    d_tip = D_P * D_L**3 / (3 * d_e * d_i) + D_P * D_L / (5 / 6 * d_g * D_B * D_H)
    d_sigma_mid = D_P * (D_L / 2) * (D_H / 2) / d_i
    return d_tip, d_sigma_mid


def solve(points, cells, ele_type, n_dim, d_thick):
    mesh = Mesh(points, cells, ele_type=ele_type)

    def clamp(p):
        return jnp.abs(p[0]) < 1e-6

    def tip(p):
        return jnp.abs(p[0] - D_L) < 1e-6

    t = np.zeros(n_dim)
    t[1] = -D_P / (D_H * d_thick)  # uniform shear traction on the end face
    zero = lambda p: 0.0  # noqa: E731
    problem = fc.Elasticity(
        mesh=mesh,
        vec=n_dim,
        dim=n_dim,
        ele_type=ele_type,
        dirichlet_bc_info=[[clamp] * n_dim, list(range(n_dim)), [zero] * n_dim],
        location_fns=[tip],
        additional_info=([t],),
    )
    n_q = problem.fes[0].num_quads
    problem.internal_vars = (np.full((len(cells), n_q), fc.D_E_PLA),)
    sol = np.asarray(solver(problem, solver_options={"umfpack_solver": {}})[0])
    b_tip = np.isclose(points[:, 0], D_L, atol=1e-6)
    d_tip = float(-sol[b_tip, 1].mean())
    reac = fc.reactions(problem, sol)
    b_clamp = np.isclose(points[:, 0], 0.0, atol=1e-6)
    d_reac_y = float(reac[b_clamp, 1].sum()) * (d_thick if n_dim == 2 else 1.0)
    sig = fc.cell_stress(problem, sol, np.full(len(cells), fc.D_E_PLA))
    cen = points[cells[:, : n_dim + 1]].mean(axis=1)
    b_mid = np.abs(cen[:, 0] - D_L / 2) < 2.0
    # Extrapolate the linear bending profile to the fibre: fit sxx vs y.
    y = cen[b_mid, 1] - D_H / 2
    sxx = sig[b_mid, :, 0, 0].mean(axis=1)
    d_slope = np.polyfit(y, sxx, 1)[0]
    return d_tip, abs(d_slope) * D_H / 2, d_reac_y, len(points)


def main():
    d_tip_th, d_sig_th = beam_theory()
    out = {"theory": {"tip_mm": d_tip_th, "sigma_mid_mpa": d_sig_th}}

    # 2-D plane stress, unit thickness scaled to the 3-D width through the load.
    rect = np.array([[0, 0], [D_L, 0], [D_L, D_H], [0, D_H]], dtype=float)
    pts2, cells2 = tri6_mesh_polygon([rect], h_fine=1.0, h_coarse=1.0, d_fine_band=1.0)
    d_tip2, d_sig2, d_r2, n2 = solve(pts2, cells2, "TRI6", 2, D_B)
    out["tri6_plane_stress"] = {
        "tip_mm": d_tip2,
        "sigma_mid_mpa": d_sig2,
        "reaction_y_n": d_r2,
        "nodes": n2,
    }

    box = trimesh.creation.box(extents=[D_L, D_H, D_B])
    box.apply_translation([D_L / 2, D_H / 2, D_B / 2])
    p4, c4 = tet_mesh_surface(box.vertices, box.faces, d_max_volume=4.0)
    p10, c10 = tet4_to_tet10(p4, c4)
    d_tip3, d_sig3, d_r3, n3 = solve(p10, c10, "TET10", 3, D_B)
    out["tet10"] = {"tip_mm": d_tip3, "sigma_mid_mpa": d_sig3, "reaction_y_n": d_r3, "nodes": n3}

    for s_key in ("tri6_plane_stress", "tet10"):
        r = out[s_key]
        r["tip_error_pct"] = 100 * (r["tip_mm"] / d_tip_th - 1)
        r["sigma_error_pct"] = 100 * (r["sigma_mid_mpa"] / d_sig_th - 1)
        r["reaction_error_pct"] = 100 * (r["reaction_y_n"] / D_P - 1)
    print(json.dumps(out, indent=2))
    (fc.FEA / "verify_elements.json").write_text(json.dumps(out, indent=2) + "\n")
    for s_key in ("tri6_plane_stress", "tet10"):
        r = out[s_key]
        # Clamping every dof at the root stiffens it slightly vs beam theory.
        assert abs(r["tip_error_pct"]) < 4, (s_key, r)
        assert abs(r["sigma_error_pct"]) < 3, (s_key, r)
        assert abs(r["reaction_error_pct"]) < 0.5, (s_key, r)
    print("element validation passed")


if __name__ == "__main__":
    main()
