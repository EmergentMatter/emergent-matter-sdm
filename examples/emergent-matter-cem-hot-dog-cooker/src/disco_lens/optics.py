"""Mirror-field optics for the disco-lens solar hot-dog cooker.

All the actual math lives here, in plain numpy (no JAX needed - nothing is
optimized by gradient, the design search is a small deterministic
coordinate-descent).

Geometry / sign conventions
---------------------------
Z-up, plate top at z = 0, dog on the +Z axis. The plate is aimed at the
sun, so incident rays travel exactly -Z in plate coordinates.

The one governing equation (law of reflection, worked in the vertical
plane through a mirror site and the axis):

    A flat mirror tilted by ``alpha`` from the plate, leaning toward the
    axis, turns a vertical ray by ``2*alpha``. For the reflected ray from
    a site at radius ``r`` (optical surface at height ``z_m``) to hit the
    NEAR WALL of the dog (radius ``r_dog``) at height ``h``:

        tan(2*alpha) = (r - r_dog) / (h - z_m)

Sites live on a square grid, so each mirror keeps its edges along the
plate axes while tilting toward the center. Its footprint along the dog is
the square's shadow in the meridional direction - the convolution of two
rects ``w*|cos az|`` and ``w*|sin az|`` - projected by ``cos(alpha)`` and
stretched by ``1/sin(2*alpha)``. Diagonal sites paint softer, wider
stripes than axis sites; the profile model carries this per site.

Aim strategy: sites are binned into radius bands one grid-pitch wide, each
band split into two interleaved parity groups (checkerboard), and each
group gets one aim height from a multi-start coordinate descent that
flattens the profile a real, hand-glued plate paints: every stripe is
convolved with its glue-scatter blur (a mirror seated ``eps`` off nominal
moves its stripe by ``2*eps*D/sin(2*alpha)`` - tens of mm for inner
sites).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from disco_lens.parameters import DiscoLensParameters, raise_if_invalid

# Assumed 1-sigma mirror seating error from hand-gluing, radians (~0.4 deg).
D_TILT_SIGMA = math.radians(0.4)


# =========================================================================
# Per-site record
# =========================================================================


@dataclass
class Site:
    n_index: int
    n_ix: int  # grid column index (signed, half-integer grid)
    n_iy: int
    d_x: float
    d_y: float
    d_radius: float
    d_azimuth: float  # radians, plate frame
    n_group: int  # aim group: radius band x checkerboard parity
    n_band: int = 0  # radius band index (monotone-aim ordering)
    # Solved per aim target:
    d_tilt: float = 0.0  # pad tilt from plate, RADIANS
    d_target_height: float = 0.0
    d_mirror_z: float = 0.0  # optical surface center height
    d_stripe_a: float = 0.0  # meridional stripe rect #1 on the dog, mm
    d_stripe_b: float = 0.0  # meridional stripe rect #2 (incl. sun blur)
    d_scatter: float = 0.0  # glue-scatter sigma along the dog, mm
    d_power: float = 0.0  # W leaving this mirror toward the dog

    @property
    def d_tilt_deg(self) -> float:
        return math.degrees(self.d_tilt)


# =========================================================================
# Elementary relations
# =========================================================================


def pad_top_z(p: DiscoLensParameters, site: Site, d_alpha: float) -> float:
    """Height of the pad top-surface center: the pad's lowest (center-
    facing) corner is held at ``d_pad_clearance`` above the plate."""
    d_drop = p.d_pad_reach * (abs(math.cos(site.d_azimuth)) + abs(math.sin(site.d_azimuth)))
    return p.d_pad_clearance + p.pad_raise(site.d_radius) + d_drop * math.sin(d_alpha)


def mirror_surface_z(p: DiscoLensParameters, site: Site, d_alpha: float) -> float:
    return pad_top_z(p, site, d_alpha) + p.d_stack_height * math.cos(d_alpha)


def solve_site(p: DiscoLensParameters, site: Site, d_h: float) -> None:
    """Fill a site's tilt, heights, stripe and power for aim height d_h.

    Tilt and mirror height depend on each other (weakly); fixed-point.
    """
    d_alpha = 0.15
    for _ in range(6):
        d_z_m = mirror_surface_z(p, site, d_alpha)
        d_alpha = 0.5 * math.atan2(site.d_radius - p.d_dog_radius, d_h - d_z_m)
    d_z_m = mirror_surface_z(p, site, d_alpha)

    d_sin = max(math.sin(2.0 * d_alpha), 1e-4)
    d_w = p.d_mirror_size
    d_c, d_s = abs(math.cos(site.d_azimuth)), abs(math.sin(site.d_azimuth))
    d_path = math.hypot(site.d_radius - p.d_dog_radius, d_h - d_z_m)
    d_blur = 2.0 * p.d_sun_half_angle * d_path / d_sin

    site.d_tilt = d_alpha
    site.d_target_height = d_h
    site.d_mirror_z = d_z_m
    site.d_stripe_a = d_w * d_c * math.cos(d_alpha) / d_sin
    # Fold the (small) sun blur into the lesser rect in quadrature.
    d_b = d_w * d_s * math.cos(d_alpha) / d_sin
    site.d_stripe_b = math.hypot(d_b, d_blur)
    site.d_scatter = 2.0 * D_TILT_SIGMA * d_path / d_sin
    site.d_power = (
        p.d_solar_irradiance * (d_w**2) * 1e-6 * math.cos(d_alpha) * p.d_mirror_reflectivity
    )


# =========================================================================
# Grid layout
# =========================================================================


def layout_sites(p: DiscoLensParameters) -> list[Site]:
    """Square grid of mirror sites, minus the central keep-out.

    Grid indices are half-integers (even cell count), so the pattern is
    exactly 4-fold rotationally symmetric and never touches the fold
    planes - the SDF assembly authors one quadrant and folds it.
    """
    n = p.n_grid_cells
    pitch = p.d_grid_pitch
    coords = [(k - n / 2.0 + 0.5) * pitch for k in range(n)]

    raw: list[tuple[int, int, float, float, float]] = []
    for ix, d_x in enumerate(coords):
        for iy, d_y in enumerate(coords):
            d_r = math.hypot(d_x, d_y)
            if d_r < p.d_min_site_radius:
                continue
            raw.append((ix, iy, d_x, d_y, d_r))

    d_r_min = min(r[4] for r in raw)
    sites: list[Site] = []
    bands: dict[int, int] = {}
    for k, (ix, iy, d_x, d_y, d_r) in enumerate(sorted(raw, key=lambda t: (t[4], t[0], t[1]))):
        n_band = int((d_r - d_r_min) / pitch)
        if n_band not in bands:  # compress to contiguous group ids
            bands[n_band] = len(bands)
        # Checkerboard parity, taken from the site's CANONICAL first-
        # quadrant representative. The SDF assembly authors quadrant 1 and
        # folds it 4x, and a 90-degree rotation of an even grid FLIPS
        # checkerboard parity - so parity must be assigned to whole fold
        # orbits or the built plate would disagree with the optics model.
        d_u = ix - n / 2.0 + 0.5
        d_v = iy - n / 2.0 + 0.5
        while not (d_u > 0.0 and d_v > 0.0):
            d_u, d_v = -d_v, d_u
        n_parity = int(round(d_u + d_v)) % 2
        n_group = 2 * bands[n_band] + n_parity
        sites.append(Site(k, ix, iy, d_x, d_y, d_r, math.atan2(d_y, d_x), n_group, bands[n_band]))

    # Compress group ids to contiguous (a band x parity cell can be empty).
    remap = {g: i for i, g in enumerate(sorted({s.n_group for s in sites}))}
    for s in sites:
        s.n_group = remap[s.n_group]
    return sites


def n_groups(sites: list[Site]) -> int:
    return max(s.n_group for s in sites) + 1


# =========================================================================
# Irradiance profile
# =========================================================================


def _trapezoid(
    z: np.ndarray, d_center: float, d_l1: float, d_l2: float, d_total: float
) -> np.ndarray:
    """Lineal power density (W/mm) of rect(L1) convolved with rect(L2),
    centered at ``d_center``, integrating to ``d_total``."""
    d_base = d_l1 + d_l2
    d_plateau = abs(d_l1 - d_l2)
    x = np.abs(z - d_center)
    ramp = np.clip((d_base / 2.0 - x) / max((d_base - d_plateau) / 2.0, 1e-9), 0.0, 1.0)
    d_height = d_total / max((d_base + d_plateau) / 2.0, 1e-9)
    return d_height * ramp


def _site_profile(site: Site, z: np.ndarray, b_scatter: bool) -> np.ndarray:
    stripe = _trapezoid(z, site.d_target_height, site.d_stripe_a, site.d_stripe_b, site.d_power)
    if b_scatter and site.d_scatter > 0.0:
        dz = float(z[1] - z[0])
        n_half = int(math.ceil(3.0 * site.d_scatter / dz))
        k = np.exp(-0.5 * (np.arange(-n_half, n_half + 1) * dz / site.d_scatter) ** 2)
        full = np.convolve(stripe, k / k.sum(), mode="full")
        stripe = full[n_half : n_half + len(z)]
    return stripe


def profile(sites: list[Site], z: np.ndarray, b_scatter: bool = False) -> np.ndarray:
    """Total lineal power density along the dog axis, W/mm."""
    out = np.zeros_like(z)
    for site in sites:
        out += _site_profile(site, z, b_scatter)
    return out


# =========================================================================
# Aim-height optimization
# =========================================================================


def _flatness(p: DiscoLensParameters, prof: np.ndarray, z: np.ndarray) -> float:
    central = (z >= p.d_dog_bottom_height + 8.0) & (z <= p.d_dog_top_height - 8.0)
    on_dog = (z >= p.d_dog_bottom_height) & (z <= p.d_dog_top_height)
    d_mean = float(np.mean(prof[central])) + 1e-12
    d_cv = float(np.std(prof[central])) / d_mean
    d_ptv = float(np.max(prof[central]) - np.min(prof[central])) / d_mean
    d_total = float(np.trapezoid(prof, z))
    d_spill = 1.0 - float(np.trapezoid(prof[on_dog], z[on_dog])) / (d_total + 1e-12)
    return d_cv + 0.4 * d_ptv + 2.0 * d_spill


def optimize_targets(p: DiscoLensParameters) -> list[Site]:
    """Choose each aim group's target height for the flattest as-built
    profile. Multi-start, multi-resolution coordinate descent; the grid is
    fixed, so each candidate move re-solves only the touched group and
    reuses cached profiles for the rest.
    """
    sites = layout_sites(p)
    n_g = n_groups(sites)
    by_group = [[s for s in sites if s.n_group == g] for g in range(n_g)]
    z = np.arange(p.d_dog_bottom_height - 40.0, p.d_dog_top_height + 40.0, 0.5)
    d_lo, d_hi = p.d_dog_bottom_height, p.d_dog_top_height

    # Occlusion guard: a beam must climb at least d_min_beam_slope, so
    # far mirrors may not aim low (their flat beams would skim the pad
    # field). Per group, the binding member is the farthest one.
    h_min = [
        max(6.0 + p.d_min_beam_slope * (s.d_radius - p.d_dog_radius) for s in by_group[g])
        for g in range(n_g)
    ]

    # A group's profile depends only on (group, aim height) - cache it so
    # every (g, h) pair is solved exactly once across sweeps and starts.
    cache_gh: dict[tuple[int, float], np.ndarray] = {}

    def group_prof(g: int, d_h: float) -> np.ndarray:
        key = (g, round(d_h * 4.0) / 4.0)
        hit = cache_gh.get(key)
        if hit is not None:
            return hit
        out = np.zeros_like(z)
        for s in by_group[g]:
            solve_site(p, s, d_h)
            out += _site_profile(s, z, b_scatter=True)
        cache_gh[key] = out
        return out

    # MONOTONE constraint: aim height must be non-decreasing with band
    # radius (inner mirrors light the bottom, outer the top). With the
    # dog seated on the hub this is structural, not aesthetic: mixing
    # steep and shallow tilts side by side lets tall pad edges eat the
    # neighbors' beams (ray-traced at 10% loss without this). Monotone
    # aims = smoothly varying tilts = stadium-seating clearance.
    band_of = [by_group[g][0].n_band for g in range(n_g)]
    # CORNER bands (entirely beyond the mid-edge radius) exist at only 4
    # azimuths. Exempt them from the monotone ladder: forcing them to the
    # extreme aims makes whatever they light AZIMUTHALLY patchy (4 blobs
    # on the dog's top, seen in the ray-traced skin map). Their beams
    # start so far out that any aim above the h_min slope floor clears
    # the field, so they may aim anywhere and fill gaps.
    d_edge_r = (p.n_grid_cells / 2.0 - 0.5) * p.d_grid_pitch + 0.1
    b_corner = [min(s.d_radius for s in by_group[g]) > d_edge_r for g in range(n_g)]
    # Corner aims are PRE-SPREAD quasi-continuously over the dog, ONE aim
    # PER FOLD ORBIT (each quadrant-1 corner site + its three rotational
    # copies), and frozen: left to the optimizer they cluster (it
    # flattens power-per-height and is blind to azimuth), and clustered
    # corner power = bright patches at four compass points on the skin.
    # ~17 orbit aims spaced ~6 mm with ~20 mm stripes = a smooth wash
    # with no per-height bumps and no azimuthal blobs; the full rings
    # flatten around it.
    corner_ids = [g for g in range(n_g) if b_corner[g]]
    q1 = sorted(
        [s for g in corner_ids for s in by_group[g] if s.d_x > 0.0 and s.d_y > 0.0],
        key=lambda s: (s.d_radius, s.d_azimuth),
    )
    aims_q1 = np.linspace(d_lo + 12.0, d_hi - 8.0, max(len(q1), 1))

    def orbit_key(s: Site) -> tuple[float, float]:
        d_x, d_y = s.d_x, s.d_y
        while not (d_x > 0.0 and d_y > 0.0):
            d_x, d_y = d_y, -d_x
        return (round(d_x, 2), round(d_y, 2))

    h_orbit = {orbit_key(s): float(a) for s, a in zip(q1, aims_q1, strict=True)}

    def h_corner(s: Site) -> float:
        # Pre-spread aims must still respect the occlusion slope floor -
        # unclamped, far corner orbits took field-skimming low aims
        # (measured 0.28 actual vs a 0.45 floor at the 506 mm plate).
        return max(h_orbit[orbit_key(s)], 6.0 + p.d_min_beam_slope * (s.d_radius - p.d_dog_radius))

    corner_prof = np.zeros_like(z)
    for g in corner_ids:
        for s in by_group[g]:
            solve_site(p, s, h_corner(s))
            corner_prof += _site_profile(s, z, b_scatter=True)

    def descend(h: list[float], candidates_of) -> tuple[list[float], float]:
        h = [0.0 if b_corner[g] else max(h[g], h_min[g]) for g in range(n_g)]
        ladder = np.array([h[g] for g in range(n_g) if not b_corner[g]])
        ladder = np.maximum.accumulate(ladder)  # monotone projection
        it = iter(ladder)
        h = [h[g] if b_corner[g] else float(next(it)) for g in range(n_g)]
        cache = [np.zeros_like(z) if b_corner[g] else group_prof(g, h[g]) for g in range(n_g)]
        total = np.sum(cache, axis=0) + corner_prof
        d_val = _flatness(p, total, z)
        for _ in range(6):
            b_moved = False
            for g in range(n_g):
                if b_corner[g]:
                    continue  # frozen at the per-orbit pre-spread aims
                d_lo_g = max(
                    [h[j] for j in range(n_g) if not b_corner[j] and band_of[j] < band_of[g]],
                    default=-1e9,
                )
                d_hi_g = min(
                    [h[j] for j in range(n_g) if not b_corner[j] and band_of[j] > band_of[g]],
                    default=1e9,
                )
                rest = total - cache[g]
                d_best_h, d_best_prof, d_score = h[g], cache[g], d_val
                for d_cand in candidates_of(h[g]):
                    if d_cand < max(h_min[g], d_lo_g) or d_cand > d_hi_g:
                        continue
                    trial = group_prof(g, float(d_cand))
                    d_try = _flatness(p, rest + trial, z)
                    if d_try < d_score:
                        d_score, d_best_h, d_best_prof = d_try, float(d_cand), trial
                if d_best_h != h[g]:
                    b_moved = True
                h[g], cache[g] = d_best_h, d_best_prof
                total = rest + cache[g]
                d_val = d_score
            if not b_moved:
                break
        return h, d_val

    coarse = np.arange(d_lo - 10.0, d_hi + 10.0, 2.0)
    rng = np.random.default_rng(0)  # seeded: runs are reproducible
    ramp_up = np.linspace(d_lo + 15.0, d_hi - 15.0, n_g)
    starts = [
        ramp_up,
        np.linspace(d_lo + 2.0, d_hi + 5.0, n_g),
        np.linspace(d_lo + 30.0, d_hi - 30.0, n_g),
    ]
    starts += [np.sort(ramp_up + rng.uniform(-20.0, 20.0, n_g)) for _ in range(3)]

    best_h, d_best = None, math.inf
    for h0 in starts:
        h, d_val = descend(list(np.clip(h0, d_lo, d_hi)), lambda _: coarse)
        for d_step in (1.0, 0.5):
            h, d_val = descend(
                h,
                lambda d_c, s=d_step: np.clip(
                    np.arange(d_c - 4.0, d_c + 4.0 + s, s), d_lo - 10.0, d_hi + 10.0
                ),
            )
        if d_val < d_best:
            d_best, best_h = d_val, list(h)

    assert best_h is not None
    for g in range(n_g):
        for s in by_group[g]:
            if b_corner[g]:
                solve_site(p, s, h_corner(s))
            else:
                solve_site(p, s, best_h[g])
    return sites


# =========================================================================
# Power / cooking estimate
# =========================================================================


@dataclass
class Design:
    p: DiscoLensParameters
    sites: list[Site]
    metrics: dict = field(default_factory=dict)


def cook_estimate(p: DiscoLensParameters, d_delivered_w: float) -> dict:
    """Lumped-capacitance warm-up of the dog. Deliberately crude - it sets
    expectations, it is not a thermal certification.

    Assumptions: dog is water-ish (rho 1050 kg/m^3, cp 3400 J/kg-K),
    absorptivity from params, combined convection+radiation film
    U = 14 W/m^2-K (calm air; any wind hurts a lot), ambient 25 C.
    Equilibrium is capped at 100 C - past that the surface just boils.
    """
    d_r = p.d_dog_radius * 1e-3
    d_l = p.d_dog_length * 1e-3
    d_mass = 1050.0 * math.pi * d_r**2 * d_l
    d_cp = 3400.0
    d_area = math.pi * (2.0 * d_r) * d_l + 2.0 * math.pi * d_r**2
    d_u = 14.0
    d_absorbed = d_delivered_w * p.d_dog_absorptivity
    d_t_amb = 25.0

    d_t_eq = min(d_t_amb + d_absorbed / (d_u * d_area), 100.0)

    d_t, d_time, d_t60 = d_t_amb, 0.0, math.nan
    d_dt = 5.0
    while d_time < 7200.0:
        d_t += d_dt * (d_absorbed - d_u * d_area * (d_t - d_t_amb)) / (d_mass * d_cp)
        d_t = min(d_t, 100.0)
        d_time += d_dt
        if math.isnan(d_t60) and d_t >= 60.0:
            d_t60 = d_time
    return {
        "d_dog_mass_g": d_mass * 1e3,
        "d_absorbed_w": d_absorbed,
        "d_equilibrium_c": d_t_eq,
        "d_minutes_to_60c": d_t60 / 60.0 if not math.isnan(d_t60) else math.inf,
    }


def design(p: DiscoLensParameters | None = None) -> Design:
    """Full design pass: lay out the grid, optimize aim, report metrics."""
    p = p or DiscoLensParameters()
    raise_if_invalid(p)
    sites = optimize_targets(p)

    z = np.arange(p.d_dog_bottom_height - 40.0, p.d_dog_top_height + 40.0, 0.5)
    prof = profile(sites, z, b_scatter=True)
    ideal = profile(sites, z)
    central = (z >= p.d_dog_bottom_height + 8.0) & (z <= p.d_dog_top_height - 8.0)
    on_dog = (z >= p.d_dog_bottom_height) & (z <= p.d_dog_top_height)

    d_delivered = float(np.trapezoid(prof[on_dog], z[on_dog]))
    d_total_out = sum(s.d_power for s in sites)

    metrics = {
        "n_mirrors": len(sites),
        "d_aperture_cm2": len(sites) * (p.d_mirror_size**2) / 100.0,
        "d_reflected_w": d_total_out,
        "d_on_dog_w": d_delivered,
        "d_spill_fraction": 1.0 - d_delivered / (d_total_out + 1e-12),
        "d_profile_cv": float(np.std(prof[central]) / np.mean(prof[central])),
        "d_profile_cv_ideal": float(np.std(ideal[central]) / np.mean(ideal[central])),
        **cook_estimate(p, d_delivered),
    }
    return Design(p, sites, metrics)
