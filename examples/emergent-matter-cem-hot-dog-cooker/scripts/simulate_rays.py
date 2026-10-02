"""Ray-trace the ASSEMBLED cooker: sun rays vs the real SDF geometry.

    uv run python scripts/simulate_rays.py

Builds the full four-quadrant tree (b_quadrant=False) with the aims from
layout/mirror_table.csv - no re-optimization, no layout math in the
loop. Sphere-traces every sun ray to its first surface, reflects it off
the SDF-gradient normal, checks pad/rim self-shadowing along the
reflected path, then intersects the dog cylinder.

Outputs img/ray_sim_3d.png, img/dog_hitmap.png and the loss chain.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from software_defined_matter import MaterialRegion, Part
from software_defined_matter.sdf.compile import make_sdf_closure

from disco_lens.design_io import MIRROR_TABLE, sites_from_table
from disco_lens.optics import profile
from disco_lens.parameters import DiscoLensParameters
from disco_lens.sdf_assembly import build_tree

ROOT = Path(__file__).resolve().parent.parent
D_RAY_PITCH = 1.0  # mm between sun rays


def trace(p, f):
    d_h = p.d_plate_half
    xs = np.arange(-d_h + 0.5, d_h, D_RAY_PITCH)
    X, Y = np.meshgrid(xs, xs)
    origins = np.stack([X.ravel(), Y.ravel(), np.full(X.size, 40.0)], axis=1)
    origins = origins[np.hypot(origins[:, 0], origins[:, 1]) > p.d_dog_radius]
    pts = jnp.asarray(origins, dtype=jnp.float32)

    def body(_, t):
        d = f(pts + t[:, None] * jnp.array([0.0, 0.0, -1.0]))
        return t + jnp.clip(d, 0.0, None)

    t = jax.lax.fori_loop(0, 100, body, jnp.zeros(pts.shape[0]))
    hits = pts + t[:, None] * jnp.array([0.0, 0.0, -1.0])
    b_hit = np.asarray(f(hits)) < 5e-3

    d_eps = 1e-3
    n = np.zeros((hits.shape[0], 3), dtype=np.float32)
    for k in range(3):
        e = np.zeros(3, dtype=np.float32)
        e[k] = d_eps
        n[:, k] = np.asarray(f(hits + e) - f(hits - e)) / (2 * d_eps)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12

    hits = np.asarray(hits)
    refl = np.tile([0.0, 0.0, -1.0], (hits.shape[0], 1)) + 2.0 * n[:, 2:3] * n

    d_a = refl[:, 0] ** 2 + refl[:, 1] ** 2
    d_b = 2.0 * (hits[:, 0] * refl[:, 0] + hits[:, 1] * refl[:, 1])
    d_c = hits[:, 0] ** 2 + hits[:, 1] ** 2 - p.d_dog_radius**2
    disc = d_b**2 - 4.0 * d_a * d_c
    with np.errstate(invalid="ignore", divide="ignore"):
        d_tc = (-d_b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * d_a)
    b_toward = (disc > 0.0) & (d_tc > 1.0) & (refl[:, 2] > 0.0)
    cyl = hits + d_tc[:, None] * refl
    b_dog = (
        b_hit & b_toward & (cyl[:, 2] >= p.d_dog_bottom_height) & (cyl[:, 2] <= p.d_dog_top_height)
    )
    b_missed = b_hit & b_toward & ~b_dog

    # Self-shadowing: march each dog-bound ray along its reflected path;
    # if it lands back on plate geometry within 60 mm it was eaten by a
    # neighboring pad edge (blockers are always near the source mirror -
    # past that, every beam climbs clear of the field).
    idx = np.flatnonzero(b_dog)
    o = jnp.asarray(hits[idx] + 0.3 * refl[idx], dtype=jnp.float32)
    d = jnp.asarray(refl[idx], dtype=jnp.float32)

    def march(_, t):
        return t + jnp.clip(f(o + t[:, None] * d), 0.0, None)

    t = np.asarray(jax.lax.fori_loop(0, 100, march, jnp.zeros(o.shape[0])))
    end = np.asarray(o) + t[:, None] * np.asarray(d)
    b_blk = (np.asarray(f(jnp.asarray(end))) < 5e-3) & (t < 60.0)
    b_blocked = np.zeros_like(b_dog)
    b_blocked[idx[b_blk]] = True
    b_dog = b_dog & ~b_blocked

    return hits, refl, cyl, b_hit, b_dog, b_missed, b_blocked


def plot_3d(p, hits, refl, cyl, b_dog) -> Path:
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(projection="3d")
    idx = np.flatnonzero(b_dog)
    rng = np.random.default_rng(1)
    idx = rng.choice(idx, size=min(320, idx.size), replace=False)
    cmap = plt.get_cmap("viridis")
    for i in idx:
        x, y, z = hits[i]
        ax.plot([x, x], [y, y], [300.0, z], color="gold", lw=0.4, alpha=0.3)
        cx, cy, cz = cyl[i]
        col = cmap((cz - p.d_dog_bottom_height) / p.d_dog_length)
        ax.plot([x, cx], [y, cy], [z, cz], color=col, lw=0.6, alpha=0.8)
    d_h = p.d_plate_half
    ax.plot(
        [-d_h, d_h, d_h, -d_h, -d_h], [-d_h, -d_h, d_h, d_h, -d_h], [0, 0, 0, 0, 0], "k-", lw=1.5
    )
    ax.plot([0, 0], [-d_h, d_h], [0, 0], "k-", lw=0.7)
    ax.plot([-d_h, d_h], [0, 0], [0, 0], "k-", lw=0.7)
    th = np.linspace(0, 2 * np.pi, 60)
    for z0 in np.linspace(p.d_dog_bottom_height, p.d_dog_top_height, 12):
        ax.plot(
            p.d_dog_radius * np.cos(th),
            p.d_dog_radius * np.sin(th),
            np.full_like(th, z0),
            color="tab:orange",
            lw=1.0,
            alpha=0.7,
        )
    for z0 in (0.0, p.d_hub_height):
        ax.plot(
            p.d_hub_diameter / 2 * np.cos(th),
            p.d_hub_diameter / 2 * np.sin(th),
            np.full_like(th, z0),
            color="gray",
            lw=1.0,
        )
    ax.set_box_aspect((1, 1, 1.0))
    ax.set_zlim(0, 320)
    ax.view_init(elev=14, azim=-55)
    ax.set_title(
        "assembled ray trace (four glued quadrants)\n"
        "(320 of the traced sun rays that end on the dog)"
    )
    fig.tight_layout()
    path = ROOT / "img" / "ray_sim_3d.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_hitmap(p, sites, cyl, b_dog, d_power_per_ray) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), width_ratios=[1.2, 1])
    az = np.degrees(np.arctan2(cyl[b_dog, 1], cyl[b_dog, 0]))
    zz = cyl[b_dog, 2]
    h = axes[0].hist2d(
        az,
        zz,
        bins=[36, 26],
        range=[[-180, 180], [p.d_dog_bottom_height, p.d_dog_top_height]],
        cmap="inferno",
    )
    fig.colorbar(h[3], ax=axes[0], label="rays per bin")
    axes[0].set_xlabel("around the dog, deg")
    axes[0].set_ylabel("height above plate, mm")
    axes[0].set_title("simulated light on the dog skin (unrolled)")

    z = np.arange(p.d_dog_bottom_height - 40.0, p.d_dog_top_height + 40.0, 0.5)
    axes[1].plot(z, profile(sites, z) * 1000.0, "k--", lw=1.2, label="designed (ideal)")
    counts, edges = np.histogram(zz, bins=52)
    centers = (edges[:-1] + edges[1:]) / 2.0
    d_w = counts * d_power_per_ray / (edges[1] - edges[0]) * 1000.0
    axes[1].step(centers, d_w, where="mid", color="tab:red", lw=1.6, label="ray-traced 3D model")
    axes[1].set_xlabel("height above plate, mm")
    axes[1].set_ylabel("lineal power, mW/mm")
    axes[1].set_title("designed vs simulated profile")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    path = ROOT / "img" / "dog_hitmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    p = DiscoLensParameters()
    sites = sites_from_table(p, MIRROR_TABLE)
    tree = build_tree(p, sites, b_quadrant=False)
    part = Part(name="assembled")
    part.add_material(MaterialRegion(material_id=1, name="petg_fdm", sdf_tree=tree))
    f = make_sdf_closure(tree, part)

    hits, refl, cyl, b_hit, b_dog, b_missed, b_blocked = trace(p, f)

    d_w_ray = p.d_solar_irradiance * (D_RAY_PITCH**2) * 1e-6 * p.d_mirror_reflectivity
    n_rays = b_hit.sum()
    print(f"rays traced:            {b_hit.size} ({D_RAY_PITCH} mm grid, dog shadow excluded)")
    print(f"hit the plate:          {n_rays}")
    print(f"blocked by pad edges:   {b_blocked.sum()} ({100.0 * b_blocked.sum() / n_rays:.1f}%)")
    print(f"reflected, missed dog:  {b_missed.sum()} ({100.0 * b_missed.sum() / n_rays:.1f}%)")
    print(
        f"reflected onto the dog: {b_dog.sum()} -> {b_dog.sum() * d_w_ray:.1f} W after mirror loss"
    )
    print(f"wrote {plot_3d(p, hits, refl, cyl, b_dog)}")
    print(f"wrote {plot_hitmap(p, sites, cyl, b_dog, d_w_ray)}")


if __name__ == "__main__":
    main()
