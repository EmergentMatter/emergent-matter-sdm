"""Render the 2-D FEA: stress maps for three slicer settings + a peak-stress summary.

Re-solves three configurations with the same code path as 01_fea_2d.py (the
sweep does not keep full fields), then draws
  * top: the installed profile, max principal stress in solid PLA, one shared
    sequential scale, deformation x10, table and handle load drawn in;
  * bottom: peak wall stress for every setting from results_2d.json against the
    long-term (creep) guide and PLA yield.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")

import importlib
import json

import matplotlib

matplotlib.use("Agg")
import fea_common as fc
import matplotlib.pyplot as plt
import numpy as np
from jax_fem.generate_mesh import Mesh
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.tri import Triangulation
from meshing import tri6_mesh_polygon

f2d = importlib.import_module("01_fea_2d")

# Reference palette (dataviz skill): blue sequential ramp 100 -> 700, text inks,
# surface, and the reserved status pair for pass / fail (always with a glyph).
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
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
TABLE_FILL, TABLE_EDGE = "#ebe9e4", "#c9c6bd"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
D_VMAX = 20.0  # MPa, top of the shared scale (anything above renders darkest)
D_CREEP = 0.25 * fc.D_YIELD_PLA
D_DEFORM = 3.0  # display magnification of the deformation (the 2-perimeter case moves 6 mm)
PANELS = [
    ("solid_100", "100 % solid"),
    ("6perim_40gyroid", "6 perimeters · 40 % gyroid"),
    ("2perim_15gyroid", "2 perimeters · 15 % gyroid"),
]
LABELS = {
    "solid_100": "100 % solid",
    "8perim_40gyroid": "8 perim · 40 % gyroid",
    "6perim_40gyroid": "6 perim · 40 % gyroid",
    "8perim_20gyroid": "8 perim · 20 % gyroid",
    "4perim_25gyroid": "4 perim · 25 % gyroid",
    "6perim_20gyroid": "6 perim · 20 % gyroid",
    "3perim_15gyroid": "3 perim · 15 % gyroid",
    "2perim_15gyroid": "2 perim · 15 % gyroid",
}


def solve_fields():
    part = fc.load_part()
    loops, field, d = f2d.profile_loops(part)
    g = f2d.geometry_refs(d)
    pts, cells = tri6_mesh_polygon(loops, f2d.D_H_FINE, f2d.D_H_COARSE, f2d.D_BAND)
    mesh = Mesh(pts, cells, ele_type="TRI6")
    probe = fc.Elasticity(
        mesh=mesh,
        vec=2,
        dim=2,
        ele_type="TRI6",
        location_fns=[lambda p: p[0] > 1e9],
        additional_info=([np.zeros(2)],),
    )
    q = np.asarray(probe.physical_quad_points)
    q3 = np.c_[q.reshape(-1, 2), np.full(q.shape[0] * q.shape[1], d["d_width"] / 2)].astype(
        np.float32
    )
    depth = -np.asarray(field(q3)).reshape(q.shape[:2])
    case = {"A_innermost": (g["cx"], g["cx"] + f2d.D_PATCH)}
    fields = {}
    for s_cfg, _ in PANELS:
        cfg = f2d.CONFIGS[s_cfg]
        if cfg["n_perim"] is None:
            e_q = np.full(depth.shape, fc.D_E_PLA)
        else:
            d_ec = f2d.infill_modulus_ratio(cfg["d_rho"])
            d_frac = 2 * cfg["d_skin"] / d["d_width"]
            e_q = np.where(
                depth < f2d.wall_thickness(cfg["n_perim"]),
                fc.D_E_PLA,
                fc.D_E_PLA * (d_frac + (1 - d_frac) * d_ec),
            )
        problem, solved = f2d.solve_config(pts, cells, e_q, g, d, case, 15 * fc.D_G)
        sol, info = solved["A_innermost"]
        s1 = fc.max_principal(fc.cell_stress(problem, sol, np.full(len(cells), fc.D_E_PLA)))
        fields[s_cfg] = (sol, s1, info)
        print(f"[render] {s_cfg}: peak {s1.max():.2f} MPa", flush=True)
    return pts, cells, loops, g, d, fields


def draw_map(ax, pts, cells, loops, g, d, sol, s1, info, s_title, cmap, norm):
    d_scale = D_DEFORM
    tri = Triangulation(
        pts[:, 0] + d_scale * sol[:, 0], pts[:, 1] + d_scale * sol[:, 1], cells[:, :3]
    )
    d_tt = d["d_table_thickness"]
    ax.add_patch(Rectangle((-90, -d_tt), 90, d_tt, fc=TABLE_FILL, ec=TABLE_EDGE, lw=0.8, zorder=0))
    ax.text(-86, -d_tt / 2, "table", color=INK2, fontsize=8.5, va="center")
    # Undeformed outline, ghosted, so the x10 deformation reads as motion.
    for loop in loops:
        closed = np.vstack([loop, loop[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    cell_s = s1.max(axis=1)
    ax.tripcolor(
        tri,
        facecolors=np.clip(cell_s, 0, D_VMAX),
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="none",
        rasterized=True,
        zorder=2,
    )
    # Contact band on the table top.
    xs = info["contact_x_mm"]
    ax.plot([min(xs), max(xs)], [0.35, 0.35], color=INK, lw=3.0, solid_capstyle="butt", zorder=3)
    ax.annotate(
        "table contact",
        ((min(xs) + max(xs)) / 2, 0.35),
        (-60, 16),
        fontsize=8,
        color=INK2,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7},
        zorder=4,
    )
    # Handle load.
    x_load = g["cx"] + f2d.D_PATCH / 2
    b_near = np.abs(pts[:, 0] - x_load) < 1.0
    y_bot = float((pts[b_near, 1] + d_scale * sol[b_near, 1]).min())
    ax.add_patch(
        FancyArrowPatch(
            (x_load, y_bot - 2),
            (x_load, y_bot - 14),
            arrowstyle="-|>",
            mutation_scale=14,
            color=INK,
            lw=1.4,
            zorder=4,
        )
    )
    ax.text(
        x_load + 2.5, y_bot - 9, "15 kg bag on\nthe handle", color=INK2, fontsize=8.5, va="center"
    )
    # Peak marker + direct label.
    i_c = int(np.argmax(cell_s))
    c_xy = (pts[cells[i_c, :3]] + d_scale * sol[cells[i_c, :3]]).mean(axis=0)
    d_peak = float(cell_s.max())
    ax.plot(*c_xy, marker="o", ms=8, mfc="none", mec=INK, mew=1.4, zorder=5)
    ok = d_peak <= D_CREEP
    xy_lab = (-62.0, -34.0)  # open space left of the loop, under the table
    ax.annotate(
        f"peak {d_peak:.1f} MPa",
        c_xy,
        xy_lab,
        fontsize=10,
        fontweight="bold",
        color=INK,
        ha="right",
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8},
        zorder=6,
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": SURFACE,
            "ec": GOOD if ok else CRITICAL,
            "lw": 1.4,
        },
    )
    ax.text(
        xy_lab[0],
        xy_lab[1] - 6.5,
        ("✓ under" if ok else "✗ over") + " the walls'\nlong-term limit",
        fontsize=8.5,
        color=INK2,
        ha="right",
        va="top",
        zorder=6,
    )
    ax.set_title(s_title, color=INK, fontsize=11.5, loc="left", pad=6)
    ax.set_aspect("equal")
    ax.set_xlim(-90, 32)
    ax.set_ylim(-112, 22)
    ax.axis("off")


def draw_bars(ax):
    res = json.loads((fc.FEA / "results_2d.json").read_text())["configs"]
    rows = [
        (
            LABELS[k],
            res[k]["cases"]["A_innermost"]["regions"]["anywhere"]["max_principal_mpa"],
            res[k]["cases"]["A_innermost"].get("infill_utilisation"),
        )
        for k in LABELS
        if k in res
    ]
    rows.sort(key=lambda r: r[1])
    y = np.arange(len(rows))[::-1]
    vals = [r[1] for r in rows]
    ax.barh(y, vals, height=0.62, color=RAMP[7], edgecolor=SURFACE, linewidth=2, zorder=2)
    for yi, (_, v, util) in zip(y, rows, strict=True):
        # 24/7 rule for both materials: walls under 25 % of yield, infill
        # under 25 % of its own strength (fea/05_summary.py uses the same).
        b_walls = v <= D_CREEP
        b_infill = util is None or util <= 0.25
        ok = b_walls and b_infill
        s_util = "" if util is None else f"   infill at {util * 100:.0f} % of its strength"
        ax.text(v + 0.6, yi, f"{v:.1f} MPa", va="center", fontsize=9.5, color=INK)
        ax.text(
            fc.D_YIELD_PLA + 3,
            yi,
            "✓" if ok else "✗",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=GOOD if ok else CRITICAL,
        )
        s_verdict = (
            "OK 24/7" if ok else ("walls too high" if not b_walls else "infill too high 24/7")
        )
        ax.text(fc.D_YIELD_PLA + 5.5, yi, s_verdict + s_util, va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=9.5, color=INK)
    for d_x, s_lab in (
        (D_CREEP, f"long-term limit ≈ {D_CREEP:.1f} MPa\n(PLA creep, 25 % of yield)"),
        (fc.D_YIELD_PLA, f"PLA yield {fc.D_YIELD_PLA:.0f} MPa"),
    ):
        ax.axvline(d_x, color=INK2, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(
            d_x + 0.8, len(rows) - 0.45, s_lab, fontsize=8.5, color=INK2, ha="left", va="bottom"
        )
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel(
        "peak tensile stress in the printed walls, 15 kg bag, handle innermost (MPa)",
        color=INK2,
        fontsize=9.5,
    )
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_ylim(-0.7, len(rows) + 0.3)


def main():
    import pickle

    p_cache = fc.FEA / "_render2d_cache.pkl"
    if p_cache.exists() and p_cache.stat().st_mtime > (fc.FEA / "results_2d.json").stat().st_mtime:
        pts, cells, loops, g, d, fields = pickle.loads(p_cache.read_bytes())
    else:
        pts, cells, loops, g, d, fields = solve_fields()
        p_cache.write_bytes(pickle.dumps((pts, cells, loops, g, d, fields)))
    cmap = LinearSegmentedColormap.from_list("seq_blue", RAMP)
    bounds = np.linspace(0, D_VMAX, 11)
    norm = BoundaryNorm(bounds, cmap.N)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(15, 11.2), facecolor=SURFACE)
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.25, 1.0],
        hspace=0.18,
        wspace=0.02,
        left=0.05,
        right=0.98,
        top=0.88,
        bottom=0.07,
    )
    fig.text(
        0.05,
        0.955,
        "Backpack hook: 2-D FEA, stress by slicer setting",
        fontsize=17,
        color=INK,
        fontweight="bold",
    )
    fig.text(
        0.05,
        0.925,
        "jax-fem plane stress, quadratic triangles (0.25 mm at the walls), table as a no-pull "
        "contact.  Stress is max principal in the solid PLA.  Deformation drawn ×3, "
        "undeformed dashed.",
        fontsize=10.5,
        color=INK2,
    )
    for i, (s_cfg, s_title) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i], facecolor=SURFACE)
        sol, s1, info = fields[s_cfg]
        draw_map(ax, pts, cells, loops, g, d, sol, s1, info, s_title, cmap, norm)
    cax = fig.add_axes([0.36, 0.515, 0.3, 0.014])
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
        ticks=[0, 5, 10, 15, 20],
    )
    cb.ax.set_xticklabels(["0", "5", "10", "15", "≥ 20 MPa"])
    cb.ax.tick_params(colors=INK2, labelsize=9, length=0)
    cb.outline.set_visible(False)
    ax_b = fig.add_axes([0.2, 0.07, 0.78, 0.36], facecolor=SURFACE)
    draw_bars(ax_b)
    r3 = json.loads((fc.FEA / "results_3d_extruded.json").read_text())["configs"]["solid_100"]
    r2h = json.loads((fc.FEA / "results_2d_h2.json").read_text())["configs"]["solid_100"]
    d_k = r3["regions"]["anywhere"]["max_principal_mpa"] / max(
        c["regions"]["anywhere"]["max_principal_mpa"] for c in r2h["cases"].values()
    )
    fig.text(
        0.05,
        0.012,
        f"A 3-D check on the same mesh size puts the solid part {100 * (d_k - 1):.1f} % higher "
        "(the 30 mm handle is narrower than the 45 mm hook).  Infill: Gibson–Ashby, E ∝ ρ², "
        "strength 0.3·σy·ρ^1.5.",
        fontsize=9,
        color=MUTED,
    )
    out = fc.FEA / "fea_2d_summary.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"[render] wrote {out}")


if __name__ == "__main__":
    main()
