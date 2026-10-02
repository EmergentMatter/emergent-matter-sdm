"""Mesh helpers: gmsh 2-D profile meshes, TetGen 3-D meshes, TET4 -> TET10."""

from __future__ import annotations

import numpy as np

# jax-fem's TET10 node order (meshio / gmsh convention, read off jax_fem.basis
# re_order): corners 0-3, then midside nodes on these edges.
TET10_EDGES = np.array([[0, 1], [1, 2], [0, 2], [0, 3], [1, 3], [2, 3]])


def tet4_to_tet10(points, cells):
    """Add a straight-edge midside node to every edge of a TET4 mesh."""
    edges = np.sort(cells[:, TET10_EDGES].reshape(-1, 2), axis=1)
    uniq, inv = np.unique(edges, axis=0, return_inverse=True)
    mids = 0.5 * (points[uniq[:, 0]] + points[uniq[:, 1]])
    new_points = np.vstack([points, mids])
    mid_ids = (len(points) + inv).reshape(-1, 6)
    return new_points, np.hstack([cells, mid_ids])


def tri6_mesh_polygon(loops, h_fine, h_coarse, d_fine_band, order=2):
    """Triangulate a closed polygon with gmsh; fine near the boundary.

    loops: list of (n, 2) arrays, first the outer boundary. Returns
    (points (n, 2), cells (m, 6)) in meshio triangle6 order.
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("profile")
    geo = gmsh.model.geo
    loop_tags, curve_tags = [], []
    for loop in loops:
        pts = [geo.addPoint(float(x), float(y), 0.0) for x, y in loop]
        lines = [geo.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
        curve_tags += lines
        loop_tags.append(geo.addCurveLoop(lines))
    geo.addPlaneSurface(loop_tags)
    geo.synchronize()
    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist, "CurvesList", curve_tags)
    gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 200)
    f_th = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
    gmsh.model.mesh.field.setNumber(f_th, "SizeMin", h_fine)
    gmsh.model.mesh.field.setNumber(f_th, "SizeMax", h_coarse)
    gmsh.model.mesh.field.setNumber(f_th, "DistMin", d_fine_band)
    gmsh.model.mesh.field.setNumber(f_th, "DistMax", d_fine_band * 2.5)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_th)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(order)
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    etype = 9 if order == 2 else 2  # gmsh: 9 = 6-node triangle, 2 = 3-node
    _, elem_nodes = gmsh.model.mesh.getElementsByType(etype)
    gmsh.finalize()
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}
    pts = coords.reshape(-1, 3)[:, :2]
    n_per = 6 if order == 2 else 3
    cells = np.array([tag_to_idx[int(t)] for t in elem_nodes]).reshape(-1, n_per)
    used = np.unique(cells)
    remap = -np.ones(len(pts), dtype=int)
    remap[used] = np.arange(len(used))
    return pts[used], remap[cells]


def tet_mesh_surface(vertices, faces, d_max_volume):
    """TetGen a closed triangle surface; returns TET4 (points, cells)."""
    import pyvista as pv
    import tetgen

    surf = pv.PolyData(vertices, np.hstack([np.full((len(faces), 1), 3), faces]).ravel())
    # Keyword quality args are silently ignored by this tetgen build; the
    # switch string is honored: p = PLC, q = radius-edge bound, a = max volume.
    # A decimated marching-cubes surface can trip TetGen's subface splitting;
    # Y (keep the surface triangulation as given) avoids that.
    for s_sw in (f"pq1.4a{d_max_volume}Q", f"pq1.6a{d_max_volume}YQ"):
        try:
            nodes, elems, *_ = tetgen.TetGen(surf).tetrahedralize(switches=s_sw)
            return np.asarray(nodes, dtype=float), np.asarray(elems, dtype=int)
        except RuntimeError as e:
            print(f"[mesh] tetgen {s_sw} failed ({e}); trying next")
    raise RuntimeError("tetgen failed with every switch set")
