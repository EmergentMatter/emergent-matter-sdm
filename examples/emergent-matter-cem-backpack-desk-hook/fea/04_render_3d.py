"""Render the extruded 3-D FEA (02_fea_3d.py) as lit surfaces + a cut-away.

Panels: solid, 6 perimeters / 40 % gyroid, and the same cut open at mid-width
so the walls, skins and infill read. Colour is max principal stress in the
solid PLA (the infill cells show 0, being a different material), on the same
sequential scale as the 2-D figure; deformation x3; the table drawn in place.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap

FEA = Path(__file__).resolve().parent
RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
TABLE, INFILL = "#dcd9d1", "#b9b7b0"
D_VMAX, D_DEFORM = 20.0, 3.0
N_BINS = 10


def grid(s_cfg):
    z = np.load(FEA / f"_fea3dx_{s_cfg}.npz")
    pts, cells = z["pts"], z["cells"][:, :4]
    g = pv.UnstructuredGrid({pv.CellType.TETRA: cells}, pts + D_DEFORM * z["sol"])
    g.cell_data["s1"] = np.clip(z["s1_cell"], 0, D_VMAX)
    g.cell_data["solid"] = (z["s1_cell"] > 0).astype(float)
    return g


def split(g):
    """Solid-PLA cells (coloured by stress) and infill cells (neutral grey)."""
    b = g.cell_data["solid"] > 0.5
    solid = g.extract_cells(np.where(b)[0]).extract_surface()
    core = g.extract_cells(np.where(~b)[0]).extract_surface() if (~b).any() else None
    return solid, core


def shot(meshes, p_out, camera):
    stepped = ListedColormap(
        LinearSegmentedColormap.from_list("b", RAMP)(np.linspace(0, 1, N_BINS))
    )
    pl = pv.Plotter(off_screen=True, window_size=(1400, 1150))
    pl.set_background(SURFACE)
    for m, kw in meshes:
        pl.add_mesh(m, **kw) if "scalars" not in kw else pl.add_mesh(
            m, cmap=stepped, clim=(0, D_VMAX), show_scalar_bar=False, **kw
        )
    pl.enable_anti_aliasing("ssaa")
    pl.camera_position = camera
    pl.screenshot(str(p_out))
    pl.close()


def main():
    res_all = json.loads((FEA / "results_3d_extruded.json").read_text())
    res = res_all["configs"]
    n_kdof = round(3 * res_all["mesh"]["nodes"] / 1000)
    d_w = 45.0
    # Near face-on across the width (like the 2-D view), tilted so the depth reads.
    look = (-22, -32, 0)
    cam = [(look[0] + 70, look[1] + 45, 215), (look[0], look[1], d_w / 2), (0, 1, 0)]
    kw_s = {
        "scalars": "s1",
        "smooth_shading": False,
        "specular": 0.0,
        "ambient": 0.62,
        "diffuse": 0.45,
    }  # true-ish colours
    panels = []
    for s_cfg, s_title in (
        ("solid_100", "100 % solid"),
        ("6perim_40gyroid", "6 perimeters · 40 % gyroid"),
    ):
        g = grid(s_cfg)
        table = pv.Box(bounds=(-100, 0, -25, 0, -2, d_w + 2))
        p_img = FEA / f"_render3d_{s_cfg}.png"
        shot([(table, {"color": TABLE, "ambient": 0.4}), (g.extract_surface(), kw_s)], p_img, cam)
        panels.append((p_img, s_title, res[s_cfg]))
    # Cut-away: keep z < mid-width and look into the cut: walls and skins in
    # colour, infill in grey.
    g = grid("6perim_40gyroid").clip(
        normal=(0, 0, 1), origin=(0, 0, d_w / 2), invert=True, crinkle=True
    )
    solid, core = split(g)
    table = pv.Box(bounds=(-100, 0, -25, 0, -2, d_w / 2))
    p_cut = FEA / "_render3d_cut.png"
    meshes = [(table, {"color": TABLE, "ambient": 0.4}), (solid, kw_s)]
    if core is not None:
        meshes.append((core, {"color": INFILL, "ambient": 0.4, "diffuse": 0.7}))
    shot(meshes, p_cut, [(look[0] + 70, look[1] + 45, 190), (look[0], look[1], d_w / 4), (0, 1, 0)])
    panels.append((p_cut, "6 perim · 40 %, cut open at mid-width", res["6perim_40gyroid"]))

    fig = plt.figure(figsize=(16, 7.6), facecolor=SURFACE)
    fig.text(
        0.03,
        0.94,
        "Backpack hook: 3-D FEA check (solid and 6 perimeters · 40 % gyroid)",
        fontsize=17,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.9,
        f"jax-fem TET10 on the extruded profile ({n_kdof}k dofs), 15 kg on a 30 mm handle, "
        "table as a no-pull contact.  Colour: max principal stress in solid PLA; infill in grey.  "
        "Deformation ×3.",
        fontsize=10.5,
        color=INK2,
    )
    for i, (p_img, s_title, r) in enumerate(panels):
        ax = fig.add_axes([0.02 + i * 0.325, 0.14, 0.31, 0.7])
        ax.imshow(mpimg.imread(p_img))
        ax.axis("off")
        ax.set_title(s_title, loc="left", color=INK, fontsize=12)
        s_peak = r.get("solid_material_peak_s1_mpa", r["regions"]["anywhere"]["max_principal_mpa"])
        s_p999 = r.get("solid_material_p99_9_s1_mpa", r["regions"]["anywhere"]["p99_9_mpa"])
        lines = [
            f"peak {s_peak:.1f} MPa   (99.9th pct {s_p999:.1f})",
            f"layer peel σzz {r['interlayer']['max_sigma_zz_tension_mpa']:.1f} MPa · "
            f"layer shear {r['interlayer']['max_interlayer_shear_mpa']:.1f} MPa",
            f"sag {r['max_displacement_mm']:.1f} mm",
        ]
        if "infill_utilisation" in r:
            lines.append(f"infill at {100 * r['infill_utilisation']:.0f} % of its strength")
        ax.text(
            0.02, -0.02, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=9.5, color=INK
        )
    cax = fig.add_axes([0.35, 0.05, 0.3, 0.018])
    norm = BoundaryNorm(np.linspace(0, D_VMAX, N_BINS + 1), N_BINS)
    cmap = ListedColormap(LinearSegmentedColormap.from_list("b", RAMP)(np.linspace(0, 1, N_BINS)))
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
        ticks=[0, 5, 10, 15, 20],
    )
    cb.ax.set_xticklabels(["0", "5", "10", "15", "≥ 20 MPa"])
    cb.ax.tick_params(colors=INK2, labelsize=9, length=0)
    cb.outline.set_visible(False)
    out = FEA / "fea_3d_summary.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"[render3d] wrote {out}")


if __name__ == "__main__":
    main()
