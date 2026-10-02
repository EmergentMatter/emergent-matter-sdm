"""Optics math checks - law of reflection, aim, uniformity, packing."""

import math

import numpy as np
import pytest

from disco_lens.optics import design, layout_sites, profile
from disco_lens.parameters import DiscoLensParameters


@pytest.fixture(scope="module")
def dsn():
    return design(DiscoLensParameters())


def test_reflection_hits_target(dsn):
    """Reflect a vertical ray off each site's mirror plane analytically and
    check it strikes the dog near-wall at the site's aim height."""
    p = dsn.p
    for s in dsn.sites[:: max(1, len(dsn.sites) // 40)]:
        d_a = s.d_tilt
        c, si = math.cos(s.d_azimuth), math.sin(s.d_azimuth)
        n = np.array([-math.sin(d_a) * c, -math.sin(d_a) * si, math.cos(d_a)])
        d_in = np.array([0.0, 0.0, -1.0])
        r = d_in - 2.0 * np.dot(d_in, n) * n
        assert r[2] > 0.0, "reflected ray must go up"
        origin = np.array([s.d_x, s.d_y, s.d_mirror_z])
        # March to the cylinder wall radius r_dog (ray heads inward).
        d_horiz = math.hypot(r[0], r[1])
        d_t = (s.d_radius - p.d_dog_radius) / d_horiz
        d_z_hit = origin[2] + d_t * r[2]
        assert abs(d_z_hit - s.d_target_height) < 0.5


def test_targets_on_the_dog(dsn):
    p = dsn.p
    for s in dsn.sites:
        assert p.d_dog_bottom_height - 12.0 <= s.d_target_height
        assert s.d_target_height <= p.d_dog_top_height + 12.0


def test_profile_flat_and_low_spill(dsn):
    assert dsn.metrics["d_profile_cv"] < 0.15
    assert dsn.metrics["d_spill_fraction"] < 0.15
    assert dsn.metrics["n_mirrors"] > 250


def test_grid_fits_and_clears():
    p = DiscoLensParameters()
    sites = layout_sites(p)
    # 4-fold symmetric: every site's 90-degree rotation is also a site.
    coords = {(round(s.d_x, 3), round(s.d_y, 3)) for s in sites}
    for x, y in coords:
        assert (round(-y, 3), round(x, 3)) in coords
    for s in sites:
        # Pad reach stays on the plate.
        assert (
            max(abs(s.d_x), abs(s.d_y)) + p.d_pad_reach <= p.d_plate_half - p.d_plate_margin + 1e-9
        )
        # Mirror clears the central keep-out.
        assert s.d_radius >= p.d_min_site_radius
    # Neighboring mirrors keep the design air gap between their edges.
    assert abs((p.d_grid_pitch - p.d_mirror_size) - p.d_mirror_gap) < 1e-9


def test_profile_integrates_to_reflected_power(dsn):
    p = dsn.p
    z = np.arange(0.0, p.d_dog_top_height + 300.0, 0.25)
    d_total = float(np.trapezoid(profile(dsn.sites, z), z))
    d_expect = sum(s.d_power for s in dsn.sites)
    assert abs(d_total - d_expect) / d_expect < 0.02
