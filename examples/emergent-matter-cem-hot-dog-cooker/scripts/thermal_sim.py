"""Will it actually cook? Transient thermal simulation of the hot dog.

    uv run python scripts/thermal_sim.py

Scenario: clear 90 F day in Austin, TX, plate aimed at the sun. Energy
chain from direct-normal irradiance through glass-mirror and geometry
losses to absorbed watts, then a 1-D radial finite-difference conduction
model of the dog (it's long and evenly lit by design, so axial gradients
are ignored) with natural-convection + radiation losses at the skin.
Two wind cases: dead calm, and a 1 m/s breeze.

Assumptions are printed with the results. The surface is capped at 100 C:
past that, water boils - in reality evaporation throttles the skin well
before boiling, so late-time skin temps here are optimistic; core times
are only mildly affected.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from disco_lens import DiscoLensParameters, design

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- scenario
D_T_AMB_C = 32.2  # 90 F
D_DNI = 900.0  # W/m^2, clear Austin summer midday, plate aimed
D_REFL_GLASS = 0.80  # second-surface silvered glass mosaic tile
D_ABS_NORMAL = 0.85  # dog skin solar absorptivity at normal incidence
D_N_SKIN = 1.4  # refractive index for the Fresnel falloff
D_EMISSIVITY = 0.90
D_WIND = (0.0, 1.0)  # m/s cases

# ------------------------------------------------------------- dog (meat)
D_RHO = 1050.0  # kg/m^3
D_CP = 3400.0  # J/kg-K (emulsified sausage, ~55-60% water)
D_K = 0.45  # W/m-K

# ---------------------------------------------------------------- air @50C
D_K_AIR = 0.028
D_NU_AIR = 1.8e-5
D_PR_AIR = 0.71
D_SIGMA = 5.670e-8


def fresnel_unpolarized(d_phi: float, n: float) -> float:
    """Reflectance at incidence d_phi (rad from surface normal)."""
    d_ci = math.cos(d_phi)
    d_st = math.sin(d_phi) / n
    if d_st >= 1.0:
        return 1.0
    d_ct = math.sqrt(1.0 - d_st * d_st)
    rs = ((d_ci - n * d_ct) / (d_ci + n * d_ct)) ** 2
    rp = ((n * d_ci - d_ct) / (n * d_ci + d_ct)) ** 2
    return 0.5 * (rs + rp)


def absorbed_power(p: DiscoLensParameters) -> dict:
    """The inefficiency chain, measured by RAY-TRACING the built .sdm.

    Every ray is traced against the real geometry, so pad self-shadowing,
    beams that miss the dog, and the pedestal are all counted by
    construction. Each landing ray's incidence angle on the CURVED dog
    surface sets its Fresnel absorption - grazing light bounces off wet
    skin instead of soaking in, and that varies ray by ray.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from simulate_rays import D_RAY_PITCH, trace
    from software_defined_matter import MaterialRegion, Part
    from software_defined_matter.sdf.compile import make_sdf_closure

    from disco_lens.design_io import MIRROR_TABLE, sites_from_table
    from disco_lens.sdf_assembly import build_tree

    # The ASSEMBLED cooker, same tree the ray sim traces - the thermal
    # answer depends on all four plates, not on one printed quadrant.
    sites = sites_from_table(p, MIRROR_TABLE)
    tree = build_tree(p, sites, b_quadrant=False)
    part = Part(name="assembled")
    part.add_material(MaterialRegion(material_id=1, name="petg_fdm", sdf_tree=tree))
    f = make_sdf_closure(tree, part)

    # trace() marches every dog-bound ray itself and has already removed
    # the self-shadowed ones from b_dog; b_blocked is what it ate.
    hits, refl, cyl, b_hit, b_dog, b_missed, b_blocked = trace(p, f)
    idx = np.flatnonzero(b_dog)

    # Per-ray incidence on the curved dog surface (normal is radial).
    d_dir = refl[idx] / np.linalg.norm(refl[idx], axis=1, keepdims=True)
    n_dog = np.column_stack([cyl[idx, 0], cyl[idx, 1], np.zeros(len(idx))]) / p.d_dog_radius
    d_cos = np.clip(-np.sum(d_dir * n_dog, axis=1), 0.0, 1.0)
    d_phi = np.arccos(d_cos)
    d_r0 = fresnel_unpolarized(0.0, D_N_SKIN)
    d_absorptance = np.array(
        [
            D_ABS_NORMAL * (1.0 - fresnel_unpolarized(float(a), D_N_SKIN)) / (1.0 - d_r0)
            for a in d_phi
        ]
    )

    d_w_ray = D_DNI * (D_RAY_PITCH**2) * 1e-6  # W per traced sun ray
    return {
        "d_collected": b_hit.sum() * d_w_ray,
        # b_blocked is a subset of the dog-bound rays that trace()
        # already removed from b_dog, so adding it back gives the
        # BEFORE-shadowing figure the report contrasts against.
        "d_after_mirror": (b_dog.sum() + b_blocked.sum()) * d_w_ray * D_REFL_GLASS,
        "d_on_skin": len(idx) * d_w_ray * D_REFL_GLASS,
        "d_absorbed": float(d_absorptance.sum()) * d_w_ray * D_REFL_GLASS,
        "n_blocked": int(b_blocked.sum()),
        "n_missed": int(b_missed.sum()),
        "d_mean_incidence_deg": float(np.degrees(d_phi.mean())),
    }


def h_natural(d_ts_c: float, d_length: float, d_dia: float) -> float:
    """Churchill-Chu vertical plate + slender-cylinder bump."""
    d_dt = max(d_ts_c - D_T_AMB_C, 0.1)
    d_t_film = (d_ts_c + D_T_AMB_C) / 2.0 + 273.15
    d_ra = 9.81 * (1.0 / d_t_film) * d_dt * d_length**3 / (D_NU_AIR * (D_NU_AIR / D_PR_AIR))
    d_nu = 0.68 + 0.670 * d_ra**0.25 / (1.0 + (0.492 / D_PR_AIR) ** (9.0 / 16.0)) ** (4.0 / 9.0)
    # Thin vertical cylinder sheds better than a flat plate.
    d_nu *= 1.0 + 1.43 * (d_length / d_dia / d_ra**0.25) ** 0.9
    return d_nu * D_K_AIR / d_length


def h_forced(d_v: float, d_dia: float) -> float:
    """Churchill-Bernstein, cylinder in crossflow."""
    d_re = d_v * d_dia / D_NU_AIR
    d_nu = 0.3 + (
        0.62
        * math.sqrt(d_re)
        * D_PR_AIR ** (1.0 / 3.0)
        / (1.0 + (0.4 / D_PR_AIR) ** (2.0 / 3.0)) ** 0.25
    ) * (1.0 + (d_re / 282000.0) ** (5.0 / 8.0)) ** (4.0 / 5.0)
    return d_nu * D_K_AIR / d_dia


def simulate(p: DiscoLensParameters, d_q_abs: float, d_wind: float, d_minutes: float = 45.0):
    """Explicit 1-D radial conduction. Returns t, T_skin, T_core (C)."""
    d_r_out = p.d_dog_radius * 1e-3
    d_len = p.d_dog_length * 1e-3
    d_a_side = math.pi * 2.0 * d_r_out * d_len
    d_a_loss = d_a_side + 2.0 * math.pi * d_r_out**2  # + ends

    n = 23
    dr = d_r_out / (n - 1)
    r = np.linspace(0.0, d_r_out, n)
    d_alpha = D_K / (D_RHO * D_CP)
    dt = 0.4 * dr**2 / (2.0 * d_alpha)
    steps = int(d_minutes * 60.0 / dt)

    T = np.full(n, D_T_AMB_C)
    out_t, out_skin, out_core = [], [], []
    for k in range(steps):
        Ts = T[-1]
        d_h = h_forced(d_wind, 2 * d_r_out) if d_wind > 0.0 else h_natural(Ts, d_len, 2 * d_r_out)
        d_hr = (
            D_EMISSIVITY
            * D_SIGMA
            * ((Ts + 273.15) ** 2 + (D_T_AMB_C + 273.15) ** 2)
            * ((Ts + 273.15) + (D_T_AMB_C + 273.15))
        )
        d_q_loss = (d_h + d_hr) * d_a_loss * (Ts - D_T_AMB_C)

        Tn = T.copy()
        # interior nodes: dT/dt = alpha * (T'' + T'/r)
        Tn[1:-1] = T[1:-1] + dt * d_alpha * (
            (T[2:] - 2 * T[1:-1] + T[:-2]) / dr**2 + (T[2:] - T[:-2]) / (2 * dr * r[1:-1])
        )
        Tn[0] = T[0] + dt * d_alpha * 4.0 * (T[1] - T[0]) / dr**2
        # surface shell energy balance
        d_shell_mass = D_RHO * d_a_side * dr / 2.0
        d_cond = D_K * d_a_side * (T[-2] - T[-1]) / dr
        Tn[-1] = T[-1] + dt * (d_q_abs - d_q_loss + d_cond) / (d_shell_mass * D_CP)
        Tn[-1] = min(Tn[-1], 100.0)  # boiling cap (evaporation, unmodeled)
        T = Tn
        if k % max(1, steps // 900) == 0:
            out_t.append(k * dt / 60.0)
            out_skin.append(T[-1])
            out_core.append(T[0])
    return np.array(out_t), np.array(out_skin), np.array(out_core)


def main() -> None:
    p = DiscoLensParameters()
    dsn = design(p)
    chain = absorbed_power(p)
    d_mass = D_RHO * math.pi * (p.d_dog_radius * 1e-3) ** 2 * p.d_dog_length * 1e-3

    print(
        f"Austin, {D_T_AMB_C:.0f} C ambient, DNI {D_DNI:.0f} W/m^2, "
        f"dog {d_mass * 1e3:.0f} g starting at ambient"
    )
    print(
        f"  sunlight onto the plate ({dsn.metrics['n_mirrors']} mirrors): "
        f"{chain['d_collected']:.1f} W"
    )
    print(
        f"  reflected onto the dog, x glass {D_REFL_GLASS:.2f}: "
        f"{chain['d_after_mirror']:.1f} W  "
        f"(spill {chain['n_missed']} rays)"
    )
    print(
        f"  after pad self-shadowing ({chain['n_blocked']} rays blocked): "
        f"{chain['d_on_skin']:.1f} W"
    )
    print(
        f"  ABSORBED by the skin (per-ray Fresnel, mean incidence "
        f"{chain['d_mean_incidence_deg']:.0f} deg): "
        f"{chain['d_absorbed']:.1f} W"
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {0.0: ("-", "dead calm"), 1.0: ("--", "1 m/s breeze")}
    results = {}
    for d_wind in D_WIND:
        t, skin, core = simulate(p, chain["d_absorbed"], d_wind)
        results[d_wind] = (t, skin, core)
        ls, s_name = styles[d_wind]
        ax.plot(t, skin, "tab:red", ls=ls, lw=2, label=f"skin, {s_name}")
        ax.plot(t, core, "tab:blue", ls=ls, lw=2, label=f"core, {s_name}")
        for d_target in (60.0, 70.0):
            idx = np.argmax(core >= d_target)
            s_when = f"{t[idx]:.0f} min" if core[idx] >= d_target else "never"
            print(
                f"  wind {d_wind:.0f} m/s: core reaches "
                f"{d_target:.0f} C at {s_when}"
                + (f", skin steady ~{skin[-1]:.0f} C" if d_target == 70.0 else "")
            )

    ax.axhline(60, color="gray", lw=0.8, ls=":")
    ax.text(0.4, 61, "60 C: hot dog is hot", fontsize=9, color="gray")
    ax.axhline(100, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("time in the sun, minutes")
    ax.set_ylabel("temperature, C")
    ax.set_title(
        f"disco-lens cook simulation - {chain['d_absorbed']:.1f} W absorbed, "
        f"90 F Austin day\n(skin capped at boiling; evaporation not modeled)"
    )
    ax.legend(fontsize=9)
    ax.set_ylim(30, 105)
    fig.tight_layout()
    path = ROOT / "img" / "cook_sim.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
