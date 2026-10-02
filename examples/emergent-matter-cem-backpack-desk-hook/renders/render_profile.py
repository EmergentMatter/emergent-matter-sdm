"""Render the installed profile (SDF slice) and a 3-D view of the STL -> renders/profile.png.

Needs the STL that build.py exports. Run from the example root:
  uv run python -B renders/render_profile.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Circle, FancyBboxPatch
from software_defined_matter import load
from software_defined_matter.grid_sampling import bind_sdf

OUT = Path(__file__).resolve().parent.parent


def main():
    part = load(OUT / "backpack_table_hook.sdm")
    d = {s_name: p.value for s_name, p in part.params.items()}
    field = bind_sdf(part.materials[0].sdf_tree, part)
    xs, ys = np.meshgrid(np.linspace(-110, 30, 700), np.linspace(-90, 25, 575))
    pts = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, d["d_width"] / 2)], -1).astype(
        np.float32
    )
    sdf = np.asarray(field(pts)).reshape(xs.shape)

    fig, (ax, ax3) = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.15, 1]})
    d_tt = d["d_table_thickness"]
    ax.add_patch(
        FancyBboxPatch(
            (-120, -d_tt),
            120,
            d_tt,
            boxstyle="round,pad=0,rounding_size=1.5",
            fc="#d9c4a0",
            ec="#8a6d3b",
            lw=1,
            label="table",
        )
    )
    ax.contourf(xs, ys, sdf, levels=[-100, 0], colors=["#3a7bd5"])
    ax.contour(xs, ys, sdf, levels=[0], colors=["#1d3f73"], linewidths=1)
    loop_cy = -d_tt - d["d_table_clearance"] - d["d_wall"] - d["d_throat"] / 2
    leg_top = loop_cy - d["d_throat"] / 2
    x_handle = -d["d_jaw_depth"] + 4
    ax.add_patch(
        Circle(
            (x_handle, leg_top + 4),
            4,
            fc="none",
            ec="#c0392b",
            lw=2,
            ls="--",
            label="handle webbing",
        )
    )
    ax.annotate(
        "load",
        (x_handle, leg_top),
        (x_handle, -84),
        ha="center",
        arrowprops={"arrowstyle": "<-", "color": "#c0392b"},
        color="#c0392b",
    )
    ax.annotate(
        "handle slides in\nover the lip",
        (-3, loop_cy + 4),
        (12, loop_cy - 2),
        fontsize=9,
        arrowprops={"arrowstyle": "->"},
    )
    ax.set_aspect("equal")
    ax.set_xlim(-110, 30)
    ax.set_ylim(-90, 25)
    ax.set_title(f"Installed profile (table {d_tt:g} mm, width {d['d_width']:g} mm)")
    ax.set_xlabel("x [mm]  (+ away from table)")
    ax.set_ylabel("y [mm]  (table top = 0)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    mesh = trimesh.load_mesh(sorted(OUT.glob("backpack_table_hook_*.stl"))[-1])
    ax3.remove()
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    v = mesh.vertices
    tri = ax3.plot_trisurf(
        v[:, 0],
        v[:, 2],
        v[:, 1],
        triangles=mesh.faces,
        color="#3a7bd5",
        edgecolor="none",
        shade=True,
        alpha=1.0,
    )
    tri.set_rasterized(True)
    ax3.set_box_aspect(np.ptp(v[:, [0, 2, 1]], axis=0))
    ax3.view_init(elev=22, azim=-58)
    ax3.set_title("3-D (width along the table edge)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("z (width)")
    ax3.set_zlabel("y")
    fig.tight_layout()
    fig.savefig(OUT / "renders" / "profile.png", dpi=130)


if __name__ == "__main__":
    main()
