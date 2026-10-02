"""Design the four-quadrant disco-lens and write every artifact.

    uv run python scripts/build.py

Outputs:
    disco_lens_quadrant.sdm   ONE printed quadrant (print four)
    disco_lens_ring.sdm       pedestal clamp ring
    layout/mirror_table.csv   one row per mirror on the ASSEMBLED plate
    img/irradiance_profile.png
    img/layout_top.png
    img/aim_map.png
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from disco_lens.harness import save_verified
from disco_lens.optics import D_TILT_SIGMA, design, n_groups, profile
from disco_lens.parameters import DiscoLensParameters
from disco_lens.sdf_assembly import build_quadrant_part, build_ring_part

ROOT = Path(__file__).resolve().parent.parent


def write_csv(dsn) -> Path:
    path = ROOT / "layout" / "mirror_table.csv"
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mirror",
                "quadrant",
                "x_mm",
                "y_mm",
                "radius_mm",
                "azimuth_deg",
                "tilt_deg",
                "aim_height_mm",
                "group",
            ]
        )
        for s in sorted(dsn.sites, key=lambda s: (s.n_ix, s.n_iy)):
            n_q = int(math.degrees(math.atan2(s.d_y, s.d_x)) % 360.0 // 90)
            w.writerow(
                [
                    s.n_index,
                    n_q + 1,
                    f"{s.d_x:.2f}",
                    f"{s.d_y:.2f}",
                    f"{s.d_radius:.2f}",
                    f"{math.degrees(s.d_azimuth):.2f}",
                    f"{s.d_tilt_deg:.3f}",
                    f"{s.d_target_height:.1f}",
                    s.n_group,
                ]
            )
    return path


def plot_profile(dsn) -> Path:
    p = dsn.p
    z = np.arange(p.d_dog_bottom_height - 40.0, p.d_dog_top_height + 40.0, 0.5)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for g in range(n_groups(dsn.sites)):
        members = [s for s in dsn.sites if s.n_group == g]
        ax.plot(z, profile(members, z) * 1000.0, lw=0.6, alpha=0.35)
    ax.plot(z, profile(dsn.sites, z) * 1000.0, "k--", lw=1.2, label="total, perfect mirrors")
    ax.plot(
        z,
        profile(dsn.sites, z, b_scatter=True) * 1000.0,
        "k",
        lw=2.4,
        label=f"total, ±{math.degrees(D_TILT_SIGMA):.1f}° glue scatter",
    )
    ax.axvspan(
        p.d_dog_bottom_height, p.d_dog_top_height, alpha=0.12, color="tab:orange", label="hot dog"
    )
    ax.set_xlabel("height above plate, mm")
    ax.set_ylabel("lineal power, mW/mm")
    ax.set_title(
        f"reflected power along the dog - CV "
        f"{dsn.metrics['d_profile_cv'] * 100:.1f}% as-built "
        f"({dsn.metrics['d_profile_cv_ideal'] * 100:.1f}% ideal)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = ROOT / "img" / "irradiance_profile.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plate_patches(ax, p, dsn, colors):
    d_h = p.d_plate_half
    ax.add_patch(Rectangle((-d_h, -d_h), 2 * d_h, 2 * d_h, fc="#e8e8e8", ec="k", lw=1))
    ax.plot([0, 0], [-d_h, d_h], "k-", lw=0.8)
    ax.plot([-d_h, d_h], [0, 0], "k-", lw=0.8)  # glue seams
    d_m = p.d_mirror_half
    for s, col in zip(dsn.sites, colors, strict=True):
        ax.add_patch(
            Rectangle((s.d_x - d_m, s.d_y - d_m), 2 * d_m, 2 * d_m, fc=col, ec="k", lw=0.2)
        )
    ax.add_patch(Circle((0, 0), p.d_ring_outer_radius, fc="#a8a8a8", ec="k"))
    ax.add_patch(Circle((0, 0), p.d_hub_diameter / 2, fc="#c8c8c8", ec="k"))
    ax.add_patch(Circle((0, 0), p.d_stick_hole_diameter / 2, fc="w", ec="k"))
    ax.set_aspect("equal")
    ax.set_xlim(-d_h - 8, d_h + 8)
    ax.set_ylim(-d_h - 8, d_h + 8)
    ax.set_xlabel("mm")


def plot_layout(dsn) -> Path:
    p = dsn.p
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    _plate_patches(ax, p, dsn, ["#9fd0ff"] * len(dsn.sites))
    ax.set_title(
        f"assembled: {dsn.metrics['n_mirrors']} mirrors "
        f"({dsn.metrics['n_mirrors'] // 4} per quadrant)\n"
        f"{p.n_grid_cells}x{p.n_grid_cells} grid, "
        f"{p.d_plate_size:.0f} mm square; lines = glue seams"
    )
    fig.tight_layout()
    path = ROOT / "img" / "layout_top.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_aim_map(dsn) -> Path:
    p = dsn.p
    fig, ax = plt.subplots(figsize=(9, 7.5))
    cmap = plt.get_cmap("plasma")
    colors = [cmap((s.d_target_height - p.d_dog_bottom_height) / p.d_dog_length) for s in dsn.sites]
    _plate_patches(ax, p, dsn, colors)
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(p.d_dog_bottom_height, p.d_dog_top_height)
    )
    cb = fig.colorbar(sm, ax=ax, shrink=0.85)
    cb.set_label(
        "where that mirror's light lands on the dog\n"
        "(mm above the plate: dark = bottom, bright = top)"
    )
    ax.set_title(f"{len(dsn.sites)} mirrors, color = aim height")
    fig.tight_layout()
    path = ROOT / "img" / "aim_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    p = DiscoLensParameters()
    dsn = design(p)

    print(p.summary())
    m = dsn.metrics
    print(f"\nmirrors:            {m['n_mirrors']} ({m['n_mirrors'] // 4} per quadrant)")
    print(f"aperture:           {m['d_aperture_cm2']:.0f} cm^2")
    print(f"reflected power:    {m['d_reflected_w']:.1f} W")
    print(f"power on the dog:   {m['d_on_dog_w']:.1f} W (spill {m['d_spill_fraction'] * 100:.0f}%)")
    print(f"profile flatness:   CV {m['d_profile_cv'] * 100:.1f}%")

    q_path = ROOT / "disco_lens_quadrant.sdm"
    save_verified(build_quadrant_part(p, dsn.sites), q_path)
    print(f"\nwrote {q_path}")
    r_path = ROOT / "disco_lens_ring.sdm"
    save_verified(build_ring_part(p), r_path)
    print(f"wrote {r_path}")
    print(f"wrote {write_csv(dsn)}")
    print(f"wrote {plot_profile(dsn)}")
    print(f"wrote {plot_layout(dsn)}")
    print(f"wrote {plot_aim_map(dsn)}")


if __name__ == "__main__":
    main()
