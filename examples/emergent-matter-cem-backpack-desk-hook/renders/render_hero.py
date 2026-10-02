"""Product renders of the print STL for the README -> renders/.

  * hero.png:  the hook clamped on a desk (table drawn at d_table_thickness,
               with its ~3.25 mm edge rounds), a bag handle in the hook.
  * print.png: the part as it goes on the bed: lying on its side, no supports.

Needs the STL that build.py exports. Run from the example root:
  uv run python -B renders/render_hero.py
"""

import json
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "renders"
BG_TOP, BG_BOT = "#f4f3ef", "#dedcd5"
PART, DESK, HANDLE = "#2a78d6", "#c9b08a", "#3b3a36"
D_DESK_R = 3.25


def desk(d_t, d_w, d_r=D_DESK_R):
    """Desk slab ending at x = 0 with rounded top and bottom edges, z across the hook."""
    ang = np.linspace(0, np.pi / 2, 16)
    top = np.c_[-d_r + d_r * np.cos(ang), -d_r + d_r * np.sin(ang)]
    bot = np.c_[-d_r + d_r * np.cos(-ang[::-1]), -d_t + d_r + d_r * np.sin(-ang[::-1])]
    prof = np.vstack([[-160, 0], top[::-1], bot[::-1], [-160, -d_t]])
    prof = prof[::-1]
    face = pv.PolyData(np.c_[prof, np.full(len(prof), -40.0)], faces=[len(prof), *range(len(prof))])
    return (
        face.extrude((0, 0, d_w + 80), capping=True)
        .triangulate()
        .compute_normals(auto_orient_normals=True)
    )


def main():
    IMG.mkdir(exist_ok=True)
    rep = json.loads((ROOT / "verification.json").read_text())
    d = rep["params"]
    stl = sorted(ROOT.glob("backpack_table_hook_pla_3dprint_*.stl"))[-1]
    part = pv.read(stl).compute_normals(split_vertices=True, feature_angle=35)
    kw_part = {
        "color": PART,
        "smooth_shading": True,
        "specular": 0.35,
        "specular_power": 30,
        "diffuse": 0.85,
        "ambient": 0.18,
    }

    # Hero: installed on the desk, a strap handle sitting in the hook.
    d_tc = d["d_table_thickness"] + d["d_table_clearance"]
    cy = -d_tc - d["d_wall"] - d["d_throat"] / 2
    y_leg = cy - d["d_throat"] / 2
    strap = pv.Tube(
        pointa=(-d["d_jaw_depth"] + 4, y_leg + 3.5, -30),
        pointb=(-d["d_jaw_depth"] + 4, y_leg + 3.5, 75),
        radius=3.5,
        n_sides=40,
    )
    pl = pv.Plotter(off_screen=True, window_size=(1800, 1200))
    pl.set_background(BG_TOP, top=BG_BOT)
    pl.add_mesh(
        desk(d["d_table_thickness"], d["d_width"]),
        color=DESK,
        smooth_shading=True,
        specular=0.1,
        ambient=0.25,
    )
    pl.add_mesh(part, **kw_part)
    pl.add_mesh(strap, color=HANDLE, smooth_shading=True, specular=0.2, ambient=0.2)
    pl.enable_ssao(radius=6)
    pl.enable_anti_aliasing("ssaa")
    pl.camera_position = [(165, 40, 190), (-25, -30, 22), (0, 1, 0)]
    pl.camera.zoom(1.25)
    pl.screenshot(str(IMG / "hero.png"))
    pl.close()

    # Print orientation: z = 0 on the bed, seen from above and in front.
    bed = pv.Plane(center=(-25, -30, -0.05), direction=(0, 0, 1), i_size=240, j_size=200)
    pl = pv.Plotter(off_screen=True, window_size=(1600, 1100))
    pl.set_background(BG_TOP, top=BG_BOT)
    pl.add_mesh(bed, color="#2e2e2c", specular=0.3, ambient=0.3)
    pl.add_mesh(part, **kw_part)
    pl.enable_ssao(radius=6)
    pl.enable_anti_aliasing("ssaa")
    pl.camera_position = [(95, -250, 230), (-26, -31, 18), (0, 0, 1)]
    pl.camera.zoom(1.05)
    pl.screenshot(str(IMG / "print.png"))
    pl.close()
    print(f"[hero] wrote {IMG / 'hero.png'} and {IMG / 'print.png'} from {stl.name}")


if __name__ == "__main__":
    main()
