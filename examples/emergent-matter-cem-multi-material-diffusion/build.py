"""Multi-material diffusion bar — a functionally graded, monolithic part.

Builds a 200 x 20 x 20 mm bar that is ONE solid piece made of two materials
(red + blue). Across the weld the two interpenetrate as a *diffusion zone*: a
stochastic dispersion of each material reaching into the other, coarse and
dense at the interface, thinning to isolated fine particles until each end is
pure. The point of the example is functionally graded multi-material CAD —
composition as a designed field, not as an assembly of two parts.

IF YOU ARE COPYING THIS AS A STARTING POINT, DON'T — USE THE TEMPLATE
-------------------------------------------------------------------
This file is ONE file on purpose. It is a worked example, meant to be read top
to bottom: the physics, the geometry it turns into, and the checks that it is
right, in the order you would think about them. Splitting that across five
modules would scatter the argument, which is the only thing this example is
for.

A real CEM should NOT look like this. `emergent-matter-sdm-cem-template` is the
source of truth for the shape of one, and it separates what this file combines:

    parameters.py     the design — frozen dataclass, derived @property,
                      validate(). Imports NOTHING from sdm-core, so a tool can
                      read the design's numbers without paying a JAX import
    sdm/trees_*.py    parameters -> SDF trees, one function per feature
    sdm/assembly.py   compose into a Part; the cem.toml entry_point
    sdm/harness.py    save, stamping metadata.generator
    sdm/cli.py        --out AND --set
    tests/            per layer, incl. field evaluation at points known by hand

Those seams are where the template's tests attach, and they are the reason it
scaffolds a CEM you can actually maintain. Start there:

    gh repo create EmergentMatter/emergent-matter-cem-<thing> --private \
      --template EmergentMatter/emergent-matter-sdm-cem-template

Everything below still obeys the template's RULES — cem.toml as the declared
surface, derived values never shipped as free params, validate() returning a
list, --out and --set, a generator cheap enough to re-run. It just declines its
FILE LAYOUT, and only because this is a demonstration.

WHAT THIS IS AND IS NOT
-----------------------
This is NOT a diffusion simulation. It uses the closed-form solution of Fick's
second law for a semi-infinite diffusion couple as a *composition target*, and
realises that target as geometry. No time stepping, no solver — an analytic
profile and a seeded scatter that matches it.

THE PHYSICS IT IS ROOTED IN
---------------------------
Two bars bonded at x=0 and held hot interdiffuse. The composition profile is
the error-function solution to Fick's second law:

    phi(x) = 1/2 * erfc( x / L ),        L = 2*sqrt(D*t)

``L`` is the PENETRATION DEPTH — the one length that sets how far the mixing
reaches. It is exposed as ``reach_mm``.

Real couples are ASYMMETRIC. The two species have different intrinsic
diffusivities D_red and D_blue, so one crosses the weld more readily than the
other — copper into brass, water into a polymer. The consequences are textbook
(Darken, Kirkendall): the faster species penetrates further, the composition
profile is lopsided about the original weld, and the lattice itself drifts.

``balance`` is that ratio, on a log scale:

    D_red / D_blue = D_CONTRAST ** balance
    L_red  = reach * (D_red/D_blue) ** 0.25      (geometric mean stays `reach`)
    L_blue = reach * (D_red/D_blue) ** -0.25

balance = 0   symmetric couple, both reach equally
balance > 0   red is the fast species — it sprawls deep into blue territory
balance < 0   blue is the fast species

The composition field is then continuous across the weld (both branches give
1/2 at x=0) but reaches differently on each side:

    phi_red(x) = 1/2 * erfc(  x / L_red )         x >= 0
    phi_red(x) = 1 - 1/2 * erfc( -x / L_blue )    x <  0

FROM COMPOSITION FIELD TO GEOMETRY
----------------------------------
A volume fraction is not printable on its own; it has to become a shape. Sites
are laid on a jittered lattice through the mixing zone. At each site the
MINORITY material is placed with probability equal to its local volume
fraction — that is the Fickian part, and it is what carries the physics. The
particle's radius is then graded between that material's min and max size by
the same fraction, which is an authoring choice on top of the physics: it buys
the coarse-at-the-weld / fine-at-the-fringe look, and it lets each material
carry its own printable feature size.

Because occupancy AND size both track the fraction, realised volume fraction
is monotone in the target but not equal to it. ``report_profile`` measures the
realised curve against the analytic one and prints both, rather than leaving
the discrepancy implied.

MONOLITHIC BY CONSTRUCTION
--------------------------
The earlier version of this example kept red and blue as two symmetric
expressions and relied on their beads never overlapping (a spacing constraint)
to stay a clean partition. A stochastic dispersion overlaps constantly, so
that construction cannot hold. This one defines a single blob and takes blue
as its exact complement inside the bar:

    red_blob = (left_block U red_particles) - blue_particles
    red      = bar AND red_blob
    blue     = bar MINUS red_blob

Inside the bar a point is in red iff red_blob is negative there and in blue iff
it is positive, so the two tile the bar exactly — no voids, no overlap, no
spacing constraint, for any particle arrangement whatsoever. Where a red and a
blue particle overlap, the subtraction gives the region to blue; that is a real
(small) bias at the weld and it is measured in the printout rather than hidden.

PARAMETERS: TWO TIERS
---------------------
``coalescence_mm`` is a live GLSL uniform. It is the ``k`` of every smooth
union/subtract, so it controls how far particles melt into each other and into
the bulk — the difference between a bag of marbles and a genuine alloy. It
scrubs in real time in the viewer.

Every other parameter changes how many spheres get emitted, which is topology.
No emitter can infer that from an already-stamped tree, so they are marked
``ui.role = "topology"`` and the .sdm carries a ``metadata.generator`` telling
the viewer how to re-run this script. Moving one of those sliders rebuilds the
part (seconds) instead of scrubbing a uniform (milliseconds).

Run:
    ../../../emergent-matter-sdm-core/.venv/bin/python build.py
    ... --set reach_mm=40 --set balance=0.6 --out /tmp/hot.sdm
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from software_defined_matter import (
    MaterialRegion,
    Param,
    Part,
    save,
    sdf_op,
)
from software_defined_matter import (
    sdf_primitive as P,
)
from software_defined_matter import (
    sdf_transform as T,
)
from software_defined_matter._meshing.bind import bind_sdf
from software_defined_matter._meshing.grid import (
    chunk_for_tree,
    eval_chunked,
    make_grid,
)
from software_defined_matter._meshing.mesh import (
    MeshCleanupConfig,
    cleanup_mesh,
    extract_mesh,
    write_mesh,
)
from software_defined_matter._meshing.types import BBox3

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Bar dimensions (mm). Long axis is X; the original weld is the plane x=0.
# ---------------------------------------------------------------------------
L_BAR, W_BAR, H_BAR = 200.0, 20.0, 20.0
HX, HY, HZ = L_BAR / 2.0, W_BAR / 2.0, H_BAR / 2.0  # 100, 10, 10

#: Widest intrinsic-diffusivity ratio D_red/D_blue reachable at |balance| = 1.
#: 16 is chosen for legibility rather than from a specific couple: it puts the
#: penetration-depth ratio at 16**0.5 = 4x across the slider, which reads
#: clearly on a 200 mm bar without one side collapsing to nothing.
D_CONTRAST = 16.0

#: Manufacturability floor (mm). A particle whose graded radius lands under
#: this is not emitted at all rather than printed as a feature no process can
#: hold. This is what leaves the fringe sparse instead of dusty.
R_FLOOR = 0.3

#: Pure-material cap at each end (mm). Particles are culled so that neither
#: material's dispersion reaches the bar ends, which keeps a grip/fixture
#: length of unmixed material and makes the gradient legible.
CAP_MM = 8.0

#: Lattice spacing (mm) at density = 1. Sets the particle budget, which the
#: ray-marched viewer pays per march step — every particle is another sphere
#: in the smooth-union fold. ~4.5 mm keeps the default part near 150 spheres.
SITE_SPACING_MM = 4.5

#: Sphere count past which the ray-marched viewer is warned. Every particle
#: is another call in the scene function's smooth-union fold, evaluated at
#: EVERY march step, so cost is linear in this and paid per pixel per frame.
#: Not a hard cap: a referee mesh is meshed once offline and can afford far
#: more than an interactive session can.
PARTICLE_BUDGET = 350

#: How far out the zone is populated, in penetration depths. erfc(1.8)/2 is
#: about 0.005, so past this the minority fraction is under half a percent and
#: the graded radius is already under R_FLOOR anyway.
ZONE_DEPTHS = 1.8

# Display colors (linear Base Color), keyed by material name.
RED = (0.85, 0.09, 0.07)
BLUE = (0.06, 0.20, 0.85)

# Meshing resolution (mm).
VOXEL = 0.2

#: Iso-level bias applied when meshing BLUE, in mm. Negative = erode.
#:
#: red and blue are an exact partition, which has a consequence for MESHING
#: that the partition check cannot see. blue = bar - red_blob is
#: `max(d_bar, -d_blob)`, so at red's boundary (d_blob = 0) blue's field is
#: `max(d_bar, 0) = 0` -- blue's ZERO-SET CONTAINS RED'S ENTIRE BOUNDARY, with
#: no interior behind it. Marching cubes both regions at iso 0 therefore emits
#: a blue surface lying exactly on every red surface: a zero-thickness sheet
#: wrapping the whole red block, which shattered into 367,831 components for
#: 356,195 vertices. In a viewer the coincident surfaces z-fight, and red
#: particles breaking the skin render as hollow rings.
#:
#: Meshing blue a quarter-voxel inside itself removes the degenerate sheet and
#: leaves a 0.05 mm gap at the interface -- a quarter of the sampling
#: resolution, well under the 0.3 mm manufacturability floor. The .sdm keeps
#: the exact partition; this is a property of the MESH, which is a lossy view
#: of the field, and it is the mesh that has to be rendered.
BLUE_ISO_BIAS_MM = -0.05


# ---------------------------------------------------------------------------
# Design variables. `bounds` is the scrub range the viewer offers; `role`
# distinguishes the one live uniform from the rebuild-on-change knobs.
# ---------------------------------------------------------------------------
PARAMS: dict[str, dict] = {
    "reach_mm": {
        "value": 34.0,
        "bounds": (3.0, 80.0),
        "unit": "mm",
        "step": 1.0,
        "role": "topology",
        "label": "diffusion distance",
        "group": "diffusion",
    },
    "balance": {
        "value": 0.0,
        "bounds": (-1.0, 1.0),
        "unit": "ratio",
        "step": 0.05,
        "role": "topology",
        "label": "diffusion balance (red <-> blue)",
        "group": "diffusion",
    },
    "sprawl": {
        "value": 0.7,
        "bounds": (0.0, 1.0),
        "unit": "ratio",
        "step": 0.05,
        "role": "topology",
        "label": "sprawl (lattice -> stochastic)",
        "group": "diffusion",
    },
    "density": {
        "value": 1.25,
        "bounds": (0.2, 2.0),
        "unit": "ratio",
        "step": 0.05,
        "role": "topology",
        "label": "particle density",
        "group": "diffusion",
    },
    "red_min_mm": {
        "value": 0.4,
        "bounds": (R_FLOOR, 6.0),
        "unit": "mm",
        "step": 0.1,
        "role": "topology",
        "label": "red particle min",
        "group": "red",
    },
    "red_max_mm": {
        "value": 4.0,
        "bounds": (0.5, 9.0),
        "unit": "mm",
        "step": 0.1,
        "role": "topology",
        "label": "red particle max",
        "group": "red",
    },
    "blue_min_mm": {
        "value": 0.4,
        "bounds": (R_FLOOR, 6.0),
        "unit": "mm",
        "step": 0.1,
        "role": "topology",
        "label": "blue particle min",
        "group": "blue",
    },
    "blue_max_mm": {
        "value": 4.0,
        "bounds": (0.5, 9.0),
        "unit": "mm",
        "step": 0.1,
        "role": "topology",
        "label": "blue particle max",
        "group": "blue",
    },
    "seed": {
        "value": 7.0,
        "bounds": (0.0, 999.0),
        "unit": "count",
        "step": 1.0,
        "role": "topology",
        "label": "scatter seed",
        "group": "diffusion",
    },
    # The one genuinely live control: k of every smooth op, emitted as a GLSL
    # uniform. Expression trees are not GLSL-emittable in sdm-core today, so a
    # live control has to be a BARE $ref — which this is.
    "coalescence_mm": {
        # 0.05 not 0.01: the template's contract check refuses a range wider
        # than 200x as "valid but unusable" — the bottom decade of a 400x
        # slider is dead travel, and 0.05 mm already reads as a hard union at
        # this scale.
        "value": 2.4,
        "bounds": (0.05, 4.0),
        "unit": "mm",
        "step": 0.01,
        "role": None,
        "label": "coalescence (melt together)",
        "group": "diffusion",
    },
}

#: The live one, referenced from every smooth op as {"$ref": ...}.
LIVE_PARAM = "coalescence_mm"


def seed_from_sdm(path: Path) -> dict[str, float]:
    """Read declared param values out of an existing .sdm at ``path``.

    OFF BY DEFAULT, and only reachable via ``--seed``. It exists for the
    second of the two contracts below, and the viewers that needed it now
    declare ``accepts`` and pass ``--set`` instead — so with seeding on by
    default the only thing that still triggered it was a HUMAN re-running this
    script in a directory where a .sdm already sat, silently inheriting
    whatever a slider session last left there. A build should give you the
    part its defaults describe.

    TWO GENERATOR CONTRACTS, and a CEM may meet either:

      command line  the viewer writes a TEMP .sdm and invokes
                    ``argv + --out <tmp> + --set k=v ...`` — every changed
                    value arrives as an argument. This is what the
                    ``generator.accepts`` declaration opts into.
      file          the viewer writes the staged values INTO the .sdm and
                    then invokes ``argv`` VERBATIM — no --set, no --out. The
                    generator is expected to READ the file it is about to
                    overwrite. sdm-view did this before it honoured
                    ``accepts``.

    Implementing only the first is why sdm-view's sliders once appeared not
    to work: it staged `coalescence_mm = 1.2`, this script rebuilt from its
    own defaults, and 2.4 came back. Nothing errored, because nothing was
    wrong from either side's point of view.

    Unknown or malformed entries are ignored rather than raising: the file is
    an input here, and a stale one should not stop a build that `--set` and
    the defaults can complete on their own.
    """
    if not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text())
        params = doc.get("params") or {}
    except (OSError, ValueError):
        return {}
    out: dict[str, float] = {}
    for name in PARAMS:
        entry = params.get(name)
        if isinstance(entry, dict) and "value" in entry:
            with contextlib.suppress(TypeError, ValueError):
                out[name] = float(entry["value"])
    return out


def resolve_params(
    overrides: dict[str, float], seeds: dict[str, float] | None = None
) -> dict[str, float]:
    """Resolve values: ``overrides`` beat ``seeds`` beat the declared defaults.

    ``seeds`` come from an existing .sdm (see :func:`seed_from_sdm`), which is
    how sdm-view hands a CEM the values a user moved.

    Deliberately does NOT clamp.

    An earlier version clamped to bounds and silently swapped an inverted
    size range. Both are worse than either accepting or refusing the input:
    the caller sends 90, gets 80, and nothing anywhere says so — the viewer
    then shows a slider position that does not describe the part it is
    looking at. Range checking belongs in :func:`validate`, which reports.
    """
    seeds = seeds or {}
    return {
        name: float(overrides.get(name, seeds.get(name, spec["value"])))
        for name, spec in PARAMS.items()
    }


def derived(p: dict[str, float]) -> dict[str, float]:
    """Everything computed from the declared params.

    These are NEVER shipped as free params. Flattening a derived value into an
    independent param is the hexafoil bug: the two desync the moment a slider
    moves and the viewer renders the inconsistent state faithfully, because
    nothing errors. They are recorded in metadata for provenance only.
    """
    d_l_red, d_l_blue = penetration_depths(p["reach_mm"], p["balance"])
    return {
        "l_red_mm": d_l_red,
        "l_blue_mm": d_l_blue,
        "matano_shift_mm": matano_shift(d_l_red, d_l_blue),
        "zone_red_mm": d_l_red * ZONE_DEPTHS,
        "zone_blue_mm": d_l_blue * ZONE_DEPTHS,
    }


def validate(p: dict[str, float]) -> list[str]:
    """Every rule the design must satisfy. Returns a list; does not raise.

    A list rather than an exception so a caller — an optimiser, a coder agent,
    a viewer handing back nine slider values at once — sees ALL the problems
    in one pass instead of fixing them one round trip at a time. ``main()`` is
    what refuses.

    The cross-param rules here mirror ``[[cem.constraints]]`` in cem.toml. The
    manifest is what a tool reads; this is what runs. Keep them in step.
    """
    errs: list[str] = []
    d = derived(p)

    # Per-param bounds. A slider cannot leave its range, but --set can, and a
    # generator is driven by --set.
    for name, spec in PARAMS.items():
        lo, hi = spec["bounds"]
        if not (lo <= p[name] <= hi):
            errs.append(f"{name}_within_bounds: {name}={p[name]:g} is outside [{lo:g}, {hi:g}]")

    # Cross-param rules — cem.toml [[cem.constraints]].
    for s_m in ("red", "blue"):
        d_lo, d_hi = p[f"{s_m}_min_mm"], p[f"{s_m}_max_mm"]
        if d_lo >= d_hi:
            errs.append(
                f"{s_m}_size_range_ordered: {s_m}_min_mm={d_lo:g} >= "
                f"{s_m}_max_mm={d_hi:g}, which is not a range"
            )
        d_depth = d[f"l_{s_m}_mm"]
        if p[f"{s_m}_max_mm"] > d_depth:
            errs.append(
                f"{s_m}_particle_fits_its_gradient: {s_m}_max_mm="
                f"{p[f'{s_m}_max_mm']:g} mm exceeds that species' penetration "
                f"depth L_{s_m}={d_depth:.2f} mm (from reach_mm={p['reach_mm']:g} "
                f"and balance={p['balance']:g}) — one particle would span the "
                "whole zone and there is no gradient left to see"
            )
    return errs


# ---------------------------------------------------------------------------
# The physics: Fick's second law for a semi-infinite couple, with asymmetric
# intrinsic diffusivities.
# ---------------------------------------------------------------------------
def penetration_depths(d_reach: float, d_balance: float) -> tuple[float, float]:
    """Per-species penetration depth L_i = 2*sqrt(D_i * t), in mm.

    The balance slider is the log-ratio of the two intrinsic diffusivities.
    Depths are normalised so their GEOMETRIC MEAN stays ``reach`` — sliding
    balance redistributes reach between the two species rather than adding
    total mixing, so the control does one thing.
    """
    d_rho = D_CONTRAST**d_balance  # D_red / D_blue
    return d_reach * d_rho**0.25, d_reach * d_rho**-0.25


def phi_red(d_x: float, d_l_red: float, d_l_blue: float) -> float:
    """Volume fraction of RED at station ``d_x``: 1 deep in red, 0 deep in blue.

    Both branches give 1/2 at the weld, so the field is continuous there; they
    differ in how fast they decay, which is the whole asymmetry.
    """
    if d_x >= 0.0:
        return 0.5 * math.erfc(d_x / d_l_red)
    return 1.0 - 0.5 * math.erfc(-d_x / d_l_blue)


def matano_shift(d_l_red: float, d_l_blue: float) -> float:
    """Net material transported across the weld, as an interface offset (mm).

    Integrating the profile, red delivers L_red/(2*sqrt(pi)) of itself into
    x>0 and blue delivers L_blue/(2*sqrt(pi)) the other way. When the species
    differ the two do not cancel, and the plane that balances them — the
    Matano interface — sits off the original weld. Reported, not built: the
    visible asymmetry is already carried by the particle field.
    """
    return (d_l_red - d_l_blue) / (2.0 * math.sqrt(math.pi))


# ---------------------------------------------------------------------------
# Composition field -> particles
# ---------------------------------------------------------------------------
def scatter(p: dict[str, float]) -> tuple[list[tuple], list[tuple]]:
    """Realise the composition profile as two particle lists.

    Returns ``(red_particles, blue_particles)``, each ``[(x, y, z, r), ...]``,
    where red particles live at x > 0 (red reaching into blue) and blue at
    x < 0. Deterministic in ``seed``.
    """
    d_l_red, d_l_blue = penetration_depths(p["reach_mm"], p["balance"])
    rng = np.random.default_rng(int(p["seed"]))
    d_s = SITE_SPACING_MM / max(p["density"], 1e-6) ** (1.0 / 3.0)

    reds: list[tuple] = []
    blues: list[tuple] = []
    d_x_limit = HX - CAP_MM

    for b_is_red, d_depth in ((True, d_l_red), (False, d_l_blue)):
        d_zone = min(d_depth * ZONE_DEPTHS, d_x_limit)
        if d_zone <= 0.0:
            continue
        d_r_min = p["red_min_mm" if b_is_red else "blue_min_mm"]
        d_r_max = p["red_max_mm" if b_is_red else "blue_max_mm"]

        # Jittered lattice over the half-zone this material reaches into.
        n_x = max(int(d_zone / d_s), 1)
        n_y = max(int(W_BAR / d_s), 1)
        n_z = max(int(H_BAR / d_s), 1)
        for i in range(n_x):
            for j in range(n_y):
                for k in range(n_z):
                    d_jit = p["sprawl"] * d_s * 0.5
                    d_x = (i + 0.5) * d_zone / n_x + rng.uniform(-d_jit, d_jit)
                    d_y = -HY + (j + 0.5) * W_BAR / n_y + rng.uniform(-d_jit, d_jit)
                    d_z = -HZ + (k + 0.5) * H_BAR / n_z + rng.uniform(-d_jit, d_jit)
                    if d_x <= 0.0:
                        continue
                    # Minority fraction, normalised to 1 at the weld.
                    d_x_signed = d_x if b_is_red else -d_x
                    d_frac = phi_red(d_x_signed, d_l_red, d_l_blue)
                    d_m = 2.0 * (d_frac if b_is_red else 1.0 - d_frac)
                    # Graded radius — an authoring choice layered on top.
                    d_r = d_r_min + (d_r_max - d_r_min) * d_m
                    # Fickian occupancy — the part carrying the physics.
                    #
                    # Sites are independent, so the particles form a BOOLEAN
                    # (Poisson) model, whose covered fraction is
                    # `1 - exp(-n*V)`, NOT `n*V`: overlap saturates coverage,
                    # and near the weld — where the particles are largest and
                    # densest — the two differ by more than 2x. Inverting the
                    # right law is what makes the realised profile track the
                    # analytic one instead of drowning the weld.
                    d_phi = 0.5 * d_m  # target fraction, this material
                    d_vol = (4.0 / 3.0) * math.pi * d_r**3
                    d_p = -(d_s**3) * math.log(max(1.0 - d_phi, 1e-6)) / d_vol
                    if rng.random() >= min(d_p, 1.0):
                        continue
                    if d_r < R_FLOOR or d_x + d_r > d_x_limit:
                        continue
                    (reds if b_is_red else blues).append((d_x_signed, d_y, d_z, d_r))
    return reds, blues


def build_regions(p: dict[str, float]):
    """Assemble the two material regions as an exact partition of the bar.

    ``blue`` is the complement of ``red_blob`` inside the bar, so the pair
    tiles the bar for ANY particle arrangement — see the module docstring.
    """
    reds, blues = scatter(p)
    k_ref = {"$ref": LIVE_PARAM}

    bar = P("box", b=[HX, HY, HZ])
    left_block = T("translate", P("box", b=[HX / 2.0, HY, HZ]), t=[-HX / 2.0, 0.0, 0.0])

    def spheres(items):
        return [T("translate", P("sphere", r=r), t=[x, y, z]) for x, y, z, r in items]

    red_bulk = sdf_op("smooth_union", [left_block, *spheres(reds)], k=k_ref)
    if blues:
        red_blob = sdf_op(
            "smooth_subtract",
            [red_bulk, sdf_op("smooth_union", spheres(blues), k=k_ref)],
            k=k_ref,
        )
    else:
        red_blob = red_bulk

    red_region = sdf_op("intersect", [bar, red_blob])
    blue_region = sdf_op("subtract", [bar, red_blob])
    return red_region, blue_region, bar, reds, blues


def build_part(p: dict[str, float]):
    red_region, blue_region, bar, reds, blues = build_regions(p)
    d_l_red, d_l_blue = penetration_depths(p["reach_mm"], p["balance"])

    params = {}
    for name, spec in PARAMS.items():
        ui = {"step": spec["step"], "group": spec["group"], "explore_bounds": list(spec["bounds"])}
        if spec["role"]:
            ui["role"] = spec["role"]
            ui["rebuild"] = True
        params[name] = Param(
            name=name,
            value=p[name],
            free=(name == LIVE_PARAM),
            bounds=spec["bounds"],
            unit=spec["unit"],
            ui=ui,
        )

    materials = [
        MaterialRegion(material_id=1, name="red", sdf_tree=red_region),
        MaterialRegion(material_id=2, name="blue", sdf_tree=blue_region),
    ]
    part = Part(
        name="multi_material_diffusion_bar",
        params=params,
        materials=materials,
        metadata={
            "version": "0.2",
            "units": "mm",
            "author": "Emergent Matter",
            "description": (
                "Functionally graded two-material bar. Composition follows the "
                "erfc solution of Fick's second law for a semi-infinite couple "
                "with asymmetric intrinsic diffusivities."
            ),
            "bbox": [[-HX, -HY, -HZ], [HX, HY, HZ]],
            # Human labels for the emitter's component segmentation, in
            # material order. INERT TODAY: sdm-core segments only when the
            # emitted scene root is a HARD union, and Part.computed_envelope()
            # joins materials with smooth_union, so a multi-material part
            # never reaches that branch and meta.json ships components: [].
            # Declared anyway — it costs nothing, it is the documented way to
            # name components, and it is what this part will need the moment
            # that heuristic accepts a smooth_union root.
            "components": [m.name for m in materials],
            "display_colors": {"red": list(RED), "blue": list(BLUE)},
            # Lets the viewer rebuild the part when a topology slider moves.
            # The tool appends --out and one --set per changed param.
            "generator": {
                # The viewer runs this synchronously on every topology
                # slider release (subprocess.run, blocking, 600 s timeout), so
                # it must do the MINIMUM that produces a .sdm: no meshing, no
                # verification sampling. Both belong to authoring, and together
                # they cost minutes — which reads in the viewer as "the input
                # was ignored".
                "argv": [
                    "../../../emergent-matter-sdm-core/.venv/bin/python",
                    "build.py",
                    "--no-mesh",
                    "--no-verify",
                ],
                "cwd": ".",
                # Which flags this CLI understands. sdm-view runs the argv
                # VERBATIM unless a document opts in here, and stages the
                # values into the .sdm instead — which only works for a
                # generator that reads its own output. Declaring this gets
                # the command-line path in both viewers, and it is why
                # seed_from_sdm() below is a fallback rather than the plan.
                "accepts": ["--out", "--set"],
            },
            "diffusion": {
                "model": "Fick erfc couple, asymmetric intrinsic diffusivities",
                "reach_mm": p["reach_mm"],
                "balance": p["balance"],
                "penetration_red_mm": d_l_red,
                "penetration_blue_mm": d_l_blue,
                "matano_shift_mm": matano_shift(d_l_red, d_l_blue),
                "n_red_particles": len(reds),
                "n_blue_particles": len(blues),
            },
        },
    )
    return part, red_region, blue_region, bar, reds, blues


# ---------------------------------------------------------------------------
# Meshing + verification
# ---------------------------------------------------------------------------
DUMMY = Part(name="_mesh")


def _eval(tree, pts, part=None):
    """Evaluate ``tree`` at ``pts``, slicing at the width sdm-core prices for it.

    ``chunk_for_tree`` derives the slice from the tree's per-point intermediate
    width, so peak memory stays roughly flat instead of scaling with the part.
    A hardcoded slice is the thing it replaces.
    """
    return np.asarray(
        eval_chunked(bind_sdf(tree, part or DUMMY), pts, chunk_size=chunk_for_tree(tree))
    )


def _eval_grid(tree, part):
    bb = BBox3(
        min_pt=np.array([-HX, -HY, -HZ], float),
        max_pt=np.array([HX, HY, HZ], float),
    ).padded(VOXEL * 2)
    pts, shape = make_grid(bb, VOXEL)
    return _eval(tree, pts, part).reshape(shape), bb


def mesh_region(tree, part, path: Path, d_iso: float = 0.0) -> int:
    d, bb = _eval_grid(tree, part)
    mesh = cleanup_mesh(extract_mesh(d, bb.min_pt, VOXEL, iso_level=d_iso), MeshCleanupConfig())
    write_mesh(mesh, str(path), "stl")
    return int(np.asarray(mesh.vertices).shape[0])


def cut_half(tree):
    """Intersect with the y >= 0 half-space for a cutaway that shows the mix."""
    keep = T("translate", P("box", b=[HX + 10.0, HY / 2.0, HZ + 10.0]), t=[0.0, HY / 2.0, 0.0])
    return sdf_op("intersect", [tree, keep])


def _sample(red_region, blue_region, bar, part, n=60):
    xs = np.linspace(-HX, HX, 6 * n)
    ys = np.linspace(-HY, HY, n)
    zs = np.linspace(-HZ, HZ, n)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    return (pts, _eval(bar, pts, part), _eval(red_region, pts, part), _eval(blue_region, pts, part))


#: Membership tolerance for the partition check. A sample landing EXACTLY on
#: the shared interface has d_red = +0.0 and d_blue = -0.0, and IEEE says
#: `-0.0 >= 0.0`, so a strict sign test books the interface itself as a void —
#: one sample in 1.2 million, reported as a failed partition. The interface is
#: a measure-zero set that both regions bound; it is not a void. Ties are
#: counted and printed separately rather than folded into either bucket, so a
#: REAL defect (which is a region, not a point) still shows up as one.
MEMBERSHIP_TOL = 1e-9


def verify_partition(pts, d_bar, d_red, d_blue) -> bool:
    """Confirm every interior point belongs to exactly one material."""
    inside = d_bar < -VOXEL
    both = int(np.count_nonzero(inside & (d_red < -MEMBERSHIP_TOL) & (d_blue < -MEMBERSHIP_TOL)))
    neither = int(np.count_nonzero(inside & (d_red > MEMBERSHIP_TOL) & (d_blue > MEMBERSHIP_TOL)))
    ties = int(
        np.count_nonzero(
            inside & ((np.abs(d_red) <= MEMBERSHIP_TOL) | (np.abs(d_blue) <= MEMBERSHIP_TOL))
        )
    )
    total = int(np.count_nonzero(inside))
    b_ok = both == 0 and neither == 0
    print(f"Monolithic check ({total:,} interior samples):")
    print(f"  overlap (both materials): {both}")
    print(f"  voids   (no material)   : {neither}")
    print(f"  on the interface (tie)  : {ties}")
    print(
        "  -> "
        + ("PASS: exactly one material everywhere" if b_ok else "FAIL: partition is not clean")
    )
    return b_ok


def report_profile(pts, d_bar, d_red, p, n_bins=37) -> None:
    """Print the REALISED red volume fraction against the analytic target.

    The occupancy draw follows the Fickian fraction but the graded radius does
    too, so the realised curve is monotone in the target rather than equal to
    it. Printing both is the honest way to show that.
    """
    d_l_red, d_l_blue = penetration_depths(p["reach_mm"], p["balance"])
    inside = d_bar < -VOXEL
    x = pts[:, 0][inside]
    is_red = (d_red < 0.0)[inside]
    edges = np.linspace(-HX, HX, n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    print("\nComposition profile — realised vs Fick erfc target:")
    print(f"  penetration depth  red {d_l_red:6.2f} mm   blue {d_l_blue:6.2f} mm")
    print(f"  Matano shift       {matano_shift(d_l_red, d_l_blue):+.2f} mm from the original weld")
    print("      x (mm)   realised            target")
    for b in range(n_bins):
        sel = idx == b
        if not np.any(sel):
            continue
        d_got = float(np.count_nonzero(is_red[sel]) / np.count_nonzero(sel))
        d_mid = 0.5 * (edges[b] + edges[b + 1])
        d_want = phi_red(d_mid, d_l_red, d_l_blue)
        bar_got = "#" * int(round(d_got * 24))
        print(f"  {d_mid:+8.1f}  {d_got:5.3f} {bar_got:<24s} {d_want:5.3f}")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--out",
        type=Path,
        default=HERE / "bar.sdm",
        help="where to write the .sdm (default: ./bar.sdm)",
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=f"override a design variable: {', '.join(PARAMS)}",
    )
    ap.add_argument(
        "--no-mesh", action="store_true", help="write the .sdm only — skip STLs and the manifest"
    )
    ap.add_argument(
        "--seed",
        action="store_true",
        help="start from the values in an existing .sdm at --out "
        "instead of the declared defaults. Off by default: a "
        "plain build is reproducible, and the viewers that "
        "needed this now pass --set instead (see the "
        "generator's `accepts`)",
    )
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip the partition check and profile report. Both "
        "sample ~1.2M points through the whole particle tree, "
        "which is ~70 s — worth it when authoring, dead weight "
        "on the viewer's rebuild path (see metadata.generator)",
    )
    args = ap.parse_args()

    overrides: dict[str, float] = {}
    for item in args.set:
        key, _, val = item.partition("=")
        key = key.strip()
        if key not in PARAMS:
            ap.error(f"unknown parameter {key!r}; known: {', '.join(PARAMS)}")
        try:
            overrides[key] = float(val)
        except ValueError:
            ap.error(f"--set {key}: {val!r} is not a number")

    seeds = seed_from_sdm(args.out) if args.seed else {}
    carried = {k: v for k, v in seeds.items() if k not in overrides and v != PARAMS[k]["value"]}
    if carried:
        print(
            f"Seeded from {args.out.name}: "
            + ", ".join(f"{k}={v:g}" for k, v in sorted(carried.items()))
        )
    p = resolve_params(overrides, seeds)
    errs = validate(p)
    if errs:
        print(f"{len(errs)} constraint(s) violated:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(2)

    part, red_region, blue_region, bar, reds, blues = build_part(p)
    d_l_red, d_l_blue = penetration_depths(p["reach_mm"], p["balance"])

    print("Diffusion couple:")
    for name in PARAMS:
        print(f"  {name:<16} {p[name]:g}")
    d_x_limit = HX - CAP_MM
    for s_m in ("red", "blue"):
        d_zone = derived(p)[f"zone_{s_m}_mm"]
        if d_zone > d_x_limit:
            print(
                f"  note: the {s_m} zone would reach {d_zone:.0f} mm but the "
                f"bar allows {d_x_limit:.0f} mm;\n"
                f"        it is truncated by the pure-material cap, so "
                f"raising reach_mm further\n"
                f"        stops changing this side."
            )
    print(f"\n  D_red/D_blue      {D_CONTRAST ** p['balance']:.2f}")
    print(f"  penetration red   {d_l_red:.2f} mm")
    print(f"  penetration blue  {d_l_blue:.2f} mm")
    n_total = len(reds) + len(blues)
    print(
        f"  particles         {len(reds)} red, {len(blues)} blue ({n_total} spheres in the scene)"
    )
    if n_total > PARTICLE_BUDGET:
        print(
            f"  ! over the {PARTICLE_BUDGET}-sphere interactive budget — the "
            "viewer evaluates every\n"
            "    one of these at every march step. Fine for meshing; lower "
            "`density`\n"
            "    or `reach_mm` if the ray-marched view crawls."
        )
    print()

    if not args.no_verify:
        pts, d_bar, d_red, d_blue = _sample(red_region, blue_region, bar, part)
        verify_partition(pts, d_bar, d_red, d_blue)
        report_profile(pts, d_bar, d_red, p)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save(part, args.out)
    print(f"\nSaved {args.out}")

    if args.no_mesh:
        return

    print(f"\nMeshing at voxel = {VOXEL} mm ...")
    jobs = [
        ("red.stl", red_region, 0.0),
        ("blue.stl", blue_region, BLUE_ISO_BIAS_MM),
        ("red_cut.stl", cut_half(red_region), 0.0),
        ("blue_cut.stl", cut_half(blue_region), BLUE_ISO_BIAS_MM),
    ]
    for fname, tree, d_iso in jobs:
        n = mesh_region(tree, part, HERE / fname, d_iso)
        print(f"  {fname:<13}: {n:,} verts" + (f"  (iso {d_iso:+g} mm)" if d_iso else ""))

    manifest = {
        "part": args.out.name,
        "voxel_mm": VOXEL,
        "solid": [
            {"name": "red", "stl": "red.stl", "color": list(RED)},
            {"name": "blue", "stl": "blue.stl", "color": list(BLUE)},
        ],
        "cut": [
            {"name": "red", "stl": "red_cut.stl", "color": list(RED)},
            {"name": "blue", "stl": "blue_cut.stl", "color": list(BLUE)},
        ],
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("\nWrote manifest.json — ready for view_blender.py")


if __name__ == "__main__":
    main()
