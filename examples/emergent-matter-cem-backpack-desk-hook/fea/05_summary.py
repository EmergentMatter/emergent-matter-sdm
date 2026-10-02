"""Turn the FEA results into safety factors per load case -> fea/summary.json.

Peak wall stress = converged 2-D peak (0.25 mm walls, worst handle position)
x the 3-D knockdown for the 30 mm handle on the 45 mm hook (3-D / 2-D peak at
the same 2 mm element size). The infill utilisation gets the same treatment
with its own knockdown. The model is linear elastic with a no-pull table
contact at one load, so the contact set does not change with the load's size
and every stress scales exactly with the bag's weight.

Load cases, each checked against Prusament PLA's printed-specimen yield:
  * short_term: the bag hangs for minutes to hours. Required factor >= 2.
  * drop_on:    the bag is let go onto the hook from rest. A suddenly applied
                load doubles the static stress. Required factor >= 1.5.
  * sustained:  the bag lives on the hook 24/7 for months. PLA creeps, so
                stay under 25 % of yield: required factor >= 4.
The infill (sparse configs) and the layer bonds are checked the same way; the
lowest factor of walls, infill and layers governs.
"""

from __future__ import annotations

import json
from pathlib import Path

FEA = Path(__file__).resolve().parent
D_YIELD = 51.0  # MPa, Prusament PLA TDS v1.1, FDM-printed ISO 527-1
# Assumption, not datasheet: layer-to-layer tensile strength of FDM PLA is
# typically 50-70 % of the in-plane value; the low end is used.
D_LAYER = 0.5 * D_YIELD
N_KG_FEA = 15  # the load every FEA run applies
CASES = {  # name: (load multiplier, required factor of safety, meaning)
    "short_term": (1.0, 2.0, "hung for minutes to hours"),
    "drop_on": (2.0, 1.5, "let go onto the hook (sudden load, x2)"),
    "sustained": (1.0, 4.0, "left on the hook 24/7 (creep: stay under 25 % of yield)"),
}
LOADS_KG = (5, 10, 15, 20)
CONFIGS = {"solid_100": "100 % solid", "6perim_40gyroid": "6 perimeters · 40 % gyroid"}


def peak(r):
    return r["regions"]["anywhere"]["max_principal_mpa"]


def main():
    r2 = json.loads((FEA / "results_2d.json").read_text())["configs"]
    r2h = json.loads((FEA / "results_2d_h2.json").read_text())["configs"]
    r2c = json.loads((FEA / "results_2d_h0.5.json").read_text())["configs"]
    r3 = json.loads((FEA / "results_3d_extruded.json").read_text())["configs"]
    out = {
        "yield_mpa": D_YIELD,
        "layer_strength_mpa_assumed": D_LAYER,
        "fea_load_kg": N_KG_FEA,
        "cases": {
            k: {"load_factor": f, "required_fos": q, "meaning": s} for k, (f, q, s) in CASES.items()
        },
        "configs": {},
    }
    for s_cfg, s_label in CONFIGS.items():
        a = r2[s_cfg]["cases"]["A_innermost"]
        s_2d = max(peak(c) for c in r2[s_cfg]["cases"].values())
        s_2d_coarse = max(peak(c) for c in r2c[s_cfg]["cases"].values())
        s_3d = r3[s_cfg].get("solid_material_peak_s1_mpa", peak(r3[s_cfg]))
        d_k = s_3d / max(peak(c) for c in r2h[s_cfg]["cases"].values())
        d_wall = s_2d * d_k  # MPa at 15 kg
        # The infill gets its own 3-D knockdown: under a narrow handle the core
        # carries more of the shear than the plane-stress model gives it.
        d_infill_util = max(c.get("infill_utilisation") or 0 for c in r2[s_cfg]["cases"].values())
        d_k_infill = None
        if d_infill_util:
            d_k_infill = r3[s_cfg]["infill_utilisation"] / max(
                c["infill_utilisation"] for c in r2h[s_cfg]["cases"].values()
            )
            d_infill_util *= d_k_infill
        lay = r3[s_cfg]["interlayer"]
        d_layer = max(lay["max_sigma_zz_tension_mpa"], lay["max_interlayer_shear_mpa"])
        rows = {}
        for n_kg in LOADS_KG:
            d_s = n_kg / N_KG_FEA
            row = {}
            for s_case, (d_f, d_req, _) in CASES.items():
                fos = {
                    "walls": D_YIELD / (d_wall * d_s * d_f),
                    "layers": D_LAYER / (d_layer * d_s * d_f),
                }
                if d_infill_util:
                    fos["infill"] = 1.0 / (d_infill_util * d_s * d_f)
                s_gov = min(fos, key=fos.get)
                row[s_case] = {
                    "fos": round(fos[s_gov], 2),
                    "governs": s_gov,
                    "pass": fos[s_gov] >= d_req,
                    "by_part": {k: round(v, 2) for k, v in fos.items()},
                }
            rows[f"{n_kg}_kg"] = row
        out["configs"][s_cfg] = {
            "label": s_label,
            "peak_wall_stress_mpa_15kg": round(d_wall, 2),
            "peak_2d_mpa": s_2d,
            "knockdown_3d": round(d_k, 3),
            "mesh_convergence_pct": round(100 * (s_2d_coarse - s_2d) / s_2d, 1),
            "infill_utilisation_15kg": round(d_infill_util, 3) if d_infill_util else None,
            "knockdown_3d_infill": round(d_k_infill, 3) if d_k_infill else None,
            "layer_peel_mpa_15kg": lay["max_sigma_zz_tension_mpa"],
            "layer_shear_mpa_15kg": lay["max_interlayer_shear_mpa"],
            "sag_mm_15kg": r3[s_cfg]["max_displacement_mm"],
            "sag_2d_mm_15kg": a["max_displacement_mm"],
            # Heaviest bag that still meets each case's required factor.
            "max_kg": {
                s_case: round(
                    N_KG_FEA * min(rows[f"{N_KG_FEA}_kg"][s_case]["by_part"].values()) / d_req, 1
                )
                for s_case, (_, d_req, _) in CASES.items()
            },
            "loads": rows,
        }
    (FEA / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    for c in out["configs"].values():
        print(
            f"[summary] {c['label']}: {c['peak_wall_stress_mpa_15kg']} MPa at 15 kg "
            f"(x{c['knockdown_3d']} 3-D), max kg {c['max_kg']}"
        )
        for s_kg, row in c["loads"].items():
            print(
                f"    {s_kg:6s} "
                + "  ".join(
                    f"{k}: {v['fos']:.2f} ({v['governs']}) {'ok' if v['pass'] else 'FAIL'}"
                    for k, v in row.items()
                )
            )


if __name__ == "__main__":
    main()
