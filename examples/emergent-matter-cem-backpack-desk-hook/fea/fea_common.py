"""Shared pieces for the backpack-hook FEA: env, material, elasticity problem.

Run everything from the example root, in its uv environment:
  uv run python -B fea/<script>.py   (or ./run_fea.sh for the whole workflow)

Units: mm, N, MPa throughout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax.numpy as jnp
import numpy as np
from jax_fem.problem import Problem

ROOT = Path(__file__).resolve().parent.parent
FEA = Path(__file__).resolve().parent
P_SDM = ROOT / "backpack_table_hook.sdm"

# Prusament PLA, FDM-printed specimens (Prusament TDS v1.1 via the
# emergent_matter_materials catalog entry `pla_3dprint`).
D_E_PLA = 2300.0  # MPa, tensile modulus
D_YIELD_PLA = 51.0  # MPa, tensile yield
D_NU_PLA = 0.35  # typical PLA; not on the TDS
D_G = 9.81


def load_part():
    """The hook .sdm as an sdm-core Part (schema already checked at build time)."""
    from software_defined_matter.model import Part

    return Part.from_dict(json.loads(P_SDM.read_text()))


def lame(d_e, d_nu):
    d_mu = d_e / (2.0 * (1.0 + d_nu))
    d_lam = d_e * d_nu / ((1.0 + d_nu) * (1.0 - 2.0 * d_nu))
    return d_lam, d_mu


class Elasticity(Problem):
    """Isotropic linear elasticity with a modulus per quadrature point.

    dim == 2 is plane stress (the effective lambda for plane stress is used).
    The modulus field arrives as ``internal_vars = (E_q,)``, shape
    (num_cells, num_quads); tractions are one constant vector per surface
    set, passed as ``additional_info = (tractions,)``.
    """

    def custom_init(self, tractions):
        self.tractions = [jnp.asarray(t, dtype=float) for t in tractions]

    def get_tensor_map(self):
        n_dim = self.dim
        d_nu = D_NU_PLA

        def stress(u_grad, d_e):
            d_lam, d_mu = lame(d_e, d_nu)
            if n_dim == 2:
                d_lam = 2.0 * d_lam * d_mu / (d_lam + 2.0 * d_mu)  # plane stress
            eps = 0.5 * (u_grad + u_grad.T)
            return d_lam * jnp.trace(eps) * jnp.eye(n_dim) + 2.0 * d_mu * eps

        return stress

    def get_surface_maps(self):
        # jax-fem adds +integral(surface_map . v) to the residual, so a physical
        # traction t enters as -t (checked on the cantilever in 00_verify).
        def make(t):
            return lambda u, x: -t

        return [make(t) for t in self.tractions]


def cell_stress(problem, sol, e_cell):
    """Per-cell stress tensor at the cell's first quadrature point, (n_cells, d, d)."""
    fe = problem.fes[0]
    cells = np.asarray(fe.cells)
    cell_u = np.asarray(sol)[cells]
    grads = np.asarray(fe.shape_grads)  # (cells, quads, nodes, dim)
    grad_u = np.einsum("cqnd,cni->cqid", grads, cell_u)  # du_i/dx_d
    eps = 0.5 * (grad_u + np.swapaxes(grad_u, -1, -2))
    d_lam, d_mu = lame(e_cell[:, None, None, None], D_NU_PLA)
    n_dim = problem.dim
    if n_dim == 2:
        d_lam = 2.0 * d_lam * d_mu / (d_lam + 2.0 * d_mu)
    tr = np.trace(eps, axis1=-2, axis2=-1)[..., None, None]
    return d_lam * tr * np.eye(n_dim) + 2.0 * d_mu * eps  # (cells, quads, d, d)


def von_mises(sig):
    n_dim = sig.shape[-1]
    if n_dim == 2:
        sxx, syy, sxy = sig[..., 0, 0], sig[..., 1, 1], sig[..., 0, 1]
        return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * sxy**2)
    s = sig - np.trace(sig, axis1=-2, axis2=-1)[..., None, None] / 3 * np.eye(3)
    return np.sqrt(1.5 * np.einsum("...ij,...ij->...", s, s))


def max_principal(sig):
    return np.linalg.eigvalsh(sig)[..., -1]


def reactions(problem, sol):
    """Force each support applies to the body, per node, (n_nodes, dim).

    At a supported dof the unconstrained residual is K u there, which is the
    support force (checked on the cantilever in 00_verify). In 2-D it is per
    unit thickness.
    """
    return np.asarray(problem.compute_residual([jnp.asarray(sol)])[0])


def assemble(problem, n_dof):
    """Full (unconstrained) stiffness K and external load f from a jax-fem Problem.

    Build the Problem with no Dirichlet conditions. jax-fem solves r(u) = 0
    with r(u) = J u + r(0); the sign of J relative to the stiffness depends on
    element orientation (a clockwise 2-D outline gives -K, positive tets give
    +K), and jax-fem's own solver does not care: -K u = -f is the same system.
    A condensed QP does care, so take alpha = sign(diag J), all one sign:
    K = alpha J, f = -alpha r(0).
    """
    import scipy.sparse as sp

    n_nodes = problem.fes[0].num_total_nodes
    res = problem.newton_update([jnp.zeros((n_nodes, problem.fes[0].vec))])
    j = sp.csr_matrix(
        (np.asarray(problem.V), (np.asarray(problem.I), np.asarray(problem.J))),
        shape=(n_dof, n_dof),
    )
    j.sum_duplicates()
    diag = j.diagonal()
    assert (diag > 0).all() or (diag < 0).all(), "mixed element orientation in the mesh"
    d_alpha = 1.0 if diag[0] > 0 else -1.0
    k = d_alpha * j
    f = -d_alpha * np.asarray(res[0]).ravel()
    assert k.dtype == np.float64 and f.dtype == np.float64, (
        "enable JAX_ENABLE_X64 before importing jax"
    )
    return k, f


class UnilateralContact:
    """Linear elasticity with a frictionless-normal, no-pull rigid support.

    dofs_c: dofs normal to the support (displacement >= 0 means lifting off).
    dofs_fix: dofs held at zero (the friction / rigid-body holds).
    The stiffness is condensed onto dofs_c (one factorisation, one solve per
    contact dof); each load is then a small bound-constrained QP:
        min 1/2 u^T S u - g^T u,  u >= 0
    whose KKT conditions are exactly the contact conditions: gap u >= 0,
    support force lam = S u - g >= 0, and u * lam = 0.
    """

    def __init__(self, k, dofs_c, dofs_fix):
        import scipy.sparse.linalg as spla

        n = k.shape[0]
        b_free = np.ones(n, dtype=bool)
        b_free[dofs_c] = False
        b_free[dofs_fix] = False
        self.i_f = np.where(b_free)[0]
        self.i_c = np.asarray(dofs_c)
        self.n = n
        k = k.tocsc()
        k_ff = k[self.i_f][:, self.i_f].tocsc()
        self.k_fc = k[self.i_f][:, self.i_c]
        self.k_cf = k[self.i_c][:, self.i_f]
        self.k_cc = k[self.i_c][:, self.i_c].toarray()
        self.lu = spla.splu(k_ff, permc_spec="COLAMD")
        self.x = self.lu.solve(self.k_fc.toarray())  # K_ff^-1 K_fc
        s = self.k_cc - self.k_cf @ self.x
        s = 0.5 * (s + s.T)
        # S carries the rigid lift/tilt modes, eigenvalue 0 in exact arithmetic;
        # round-off can leave them slightly negative, making the QP unbounded.
        # A 1e-9 relative diagonal dominates that round-off and moves the
        # physical answer negligibly (the callers check equilibrium).
        self.d_reg = 1e-9 * float(np.abs(np.diag(s)).max())
        self.s = s + self.d_reg * np.eye(len(s))
        self.d_min_eig = float(np.linalg.eigvalsh(s)[0] / np.abs(np.diag(s)).max())

    def solve(self, f):
        from scipy.optimize import minimize

        f_f, f_c = f[self.i_f], f[self.i_c]
        u_f0 = self.lu.solve(f_f)
        g = f_c - self.k_cf @ u_f0
        s = self.s
        d_scale = np.abs(g).max()
        gs = g / d_scale

        def obj(u):
            su = s @ u
            return 0.5 * u @ su - gs @ u, su - gs

        r = minimize(
            obj,
            np.zeros(len(g)),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * len(g),
            options={"ftol": 1e-16, "gtol": 1e-13, "maxiter": 50000},
        )
        u_c = r.x * d_scale
        # Exact finish: L-BFGS-B only guesses the held set on this ill-conditioned
        # S. Block principal pivoting: solve with the held set pinned, then move
        # lifted nodes that would sink into the held set and held nodes that
        # would pull out of it, until neither happens.
        b_held = u_c <= 1e-6 * max(np.abs(u_c).max(), 1e-30)
        n_pivots = 0
        for n_pivots in range(200):  # noqa: B007 - read after the loop as the round count
            i_free = np.where(~b_held)[0]
            u_c = np.zeros(len(g))
            if len(i_free):
                u_c[i_free] = np.linalg.solve(self.s[np.ix_(i_free, i_free)], g[i_free])
            lam = self.s @ u_c - g
            d_tol = 1e-10 * np.abs(g).max()
            b_sink = ~b_held & (u_c < 0)
            b_pull = b_held & (lam < -d_tol)
            if not b_sink.any() and not b_pull.any():
                break
            b_held = (b_held | b_sink) & ~b_pull
        else:
            raise RuntimeError("contact pivoting did not converge")
        u_c[b_held] = 0.0
        lam = s @ u_c - g
        u = np.zeros(self.n)
        u[self.i_c] = u_c
        u[self.i_f] = u_f0 - self.x @ u_c
        d_lam_max = float(np.abs(lam).max())
        info = {
            "qp_warmstart_iterations": int(r.nit),
            "pivot_rounds": int(n_pivots),
            "condensed_min_eig_rel": self.d_min_eig,
            "held": int(b_held.sum()),
            # Complementarity: the worst pull (negative support force), and the
            # worst force at a node that has lifted, relative to the peak force.
            "min_support_force_rel": float(lam.min() / d_lam_max),
            "force_at_lifted_rel": float(np.abs(lam[~b_held]).max() / d_lam_max)
            if (~b_held).any()
            else 0.0,
        }
        return u, lam, u_c, info


# Quadratic boundary entities as (corner..., midside...) local node indices.
_TRI6_EDGES = np.array([[0, 1, 3], [1, 2, 4], [2, 0, 5]])
_TET10_FACES = np.array(
    [[0, 1, 2, 4, 5, 6], [0, 1, 3, 4, 8, 7], [0, 2, 3, 6, 9, 7], [1, 2, 3, 5, 9, 8]]
)


def boundary_entities(points, cells):
    """Boundary edges (TRI6) / faces (TET10) as (corner..., midside...) node ids."""
    n_dim = points.shape[1]
    ents = _TRI6_EDGES if n_dim == 2 else _TET10_FACES
    n_corner = 2 if n_dim == 2 else 3
    allf = cells[:, ents].reshape(-1, ents.shape[1])
    key = np.sort(allf[:, :n_corner], axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return allf[cnt[inv] == 1]


def face_load_3d(points, sel, traction):
    """Consistent TET10-face load (midsides 1/3 each of t*A) on the faces sel."""
    f = np.zeros((len(points), 3))
    a, b, c = (points[sel[:, j]] for j in range(3))
    d_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    for j in (3, 4, 5):
        np.add.at(f, sel[:, j], (d_area / 3)[:, None] * np.asarray(traction, dtype=float))
    return f.ravel(), float(d_area.sum())


def patch_load(points, cells, b_node, traction):
    """Consistent nodal load of a uniform traction on the boundary entities
    whose nodes all satisfy b_node. TRI6 edges: 1/6, 1/6, 2/3 of t*L; TET10
    faces: corners 0, midsides 1/3 each of t*A. Returns (n_nodes*dim,).
    """
    n_dim = points.shape[1]
    ents = _TRI6_EDGES if n_dim == 2 else _TET10_FACES
    n_corner = 2 if n_dim == 2 else 3
    allf = cells[:, ents].reshape(-1, ents.shape[1])
    key = np.sort(allf[:, :n_corner], axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bdy = allf[cnt[inv] == 1]
    sel = bdy[b_node[bdy].all(axis=1)]
    f = np.zeros((len(points), n_dim))
    t = np.asarray(traction, dtype=float)
    if n_dim == 2:
        d_len = np.linalg.norm(points[sel[:, 1]] - points[sel[:, 0]], axis=1)
        for j, w in ((0, 1 / 6), (1, 1 / 6), (2, 2 / 3)):
            np.add.at(f, sel[:, j], (w * d_len)[:, None] * t)
        d_meas = float(d_len.sum())
    else:
        a, b, c = (points[sel[:, j]] for j in range(3))
        d_area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        for j in (3, 4, 5):
            np.add.at(f, sel[:, j], (d_area / 3)[:, None] * t)
        d_meas = float(d_area.sum())
    return f.ravel(), d_meas
