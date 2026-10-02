"""Design parameters for the disco-lens solar hot-dog cooker.

Single source of truth for every dimension in the design, following the
org frozen-dataclass pattern (cf. hexafoil ``HexafoilParameters``).

Coordinate frame: Z-up, plate top face at z = 0, stick along +Z through the
origin. The plate is aimed so the sun is on the +Z axis (incident rays travel
-Z). All lengths mm, all angles in the public API degrees unless a name says
otherwise (internal optics math is radians).

Layout: a SQUARE plate carrying a square grid of individually tilted mirror
pads. No pockets - each mirror seat is an open tilted pad with an L-shaped
locating corner (two low stubs); glue does the holding. This is the
maximum-density / minimum-material layout: pitch is mirror + clearance +
one stub, and the plate corners carry mirrors that a disc layout wastes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DiscoLensParameters:
    # ------------------------------------------------------------------
    # Plate / print
    # ------------------------------------------------------------------
    # The ASSEMBLED square: four identical quadrants glued edge to
    # edge. One PRINTED quadrant is half of this, sized so a quadrant
    # and its sight tab fit a Prusa 250x210 bed with slicer margins.
    d_plate_size: float = 410.4
    d_base_thickness: float = 3.0  # flat slab under everything
    d_plate_margin: float = 0.25  # pad keep-out from the plate edge
    d_plate_corner_radius: float = 8.0  # rounded plan corners
    d_rim_height: float = 12.0  # stiffening wall above the plate top
    d_rim_thickness: float = 2.4  # 6 perimeters of 0.4

    # ------------------------------------------------------------------
    # Mirrors + seats (glass mosaic tiles, glued flat onto bare pads -
    # no locating walls: they cost material and block ~4% of the light)
    # ------------------------------------------------------------------
    d_mirror_size: float = 10.05  # measured square mirror edge
    d_mirror_thickness: float = 2.0  # ASSUMED tile thickness - check yours
    d_glue_gap: float = 0.3  # adhesive layer under the mirror
    d_mirror_gap: float = 1.0  # clear space between mirror edges
    d_pad_extra: float = 0.2  # pad margin beyond the mirror edge
    # FUSED pad collars (Rob's call): the bare 10.45 mm pads
    # left 0.6 mm canyons - thousands of tiny print islands per layer.
    # Each pad gets a wider COLLAR (mirror + d_fuse_ledge per side) with
    # a HORIZONTAL top d_fuse_drop below the pad-plane center: collars
    # overlap edge neighbors by 0.4 mm in plan and weld the field into
    # one body; the recess keeps them clear of the glass, and horizontal
    # tops keep every junction parallel or perpendicular (tilted tops
    # knifed against the slab top and their seam twins). Mirror pitch,
    # the 1.0 mm mirror air gap, and the pad-outline placement guide are
    # all unchanged.
    d_fuse_ledge: float = 0.7  # collar margin beyond the mirror
    d_fuse_drop: float = 1.5  # collar top below the pad plane
    d_pad_clearance: float = 0.4  # lowest pad corner above plate top
    # STADIUM SEATING. Perimeter pads ride a ramp up to this much
    # extra height at the outermost sites: raising a mirror lifts its
    # whole beam, so it clears the inboard neighbors' raised edges
    # (less self-shadowing) for a little plastic. It is how blocking is
    # bought back at hub 60, and Rob allows 20 mm of mirror height
    # differential. The ramp is QUADRATIC (see pad_raise), so the
    # gradient goes where the steep pads are. Ray-traced: 6 linear ->
    # 16.4% blocked / 68.1 W absorbed; 20 quadratic -> 12.7% / 71.7 W.
    d_perimeter_raise: float = 20.0

    # ------------------------------------------------------------------
    # Corner sun-sights: tall bosses with a narrow through-bore on tabs
    # outside the mirror field (the Prusa bed is 250 wide, the plate 210,
    # so the tabs use the spare X). Sunlight passes the bore only within
    # ~atan(hole / (height + base)) of dead-on: a bright dot below each
    # corner means the plate is aimed.
    # ------------------------------------------------------------------
    d_sight_tab_length: float = 18.0  # tab reach beyond the plate edge, +X
    d_sight_tab_width: float = 26.0  # one boss per tab
    d_sight_tab_root: float = 9.0  # tab reach INTO the plate
    d_sight_tab_corner_radius: float = 5.0
    d_sight_boss_diameter: float = 9.0
    d_sight_boss_height: float = 28.0  # above plate top (16 above block)
    d_sight_hole_diameter: float = 1.4  # narrower bore keeps the cutoff
    # tight despite the shorter tower
    d_sight_boss_fillet: float = 3.0  # conical skirt at the boss root

    # ------------------------------------------------------------------
    # Hub + stick
    # ------------------------------------------------------------------
    # The dog SEATS on the hub, so hub height sets the dog's position.
    # Swept 18-80: low = steep tilts = pads shade each other (12% loss),
    # high = grazing skin incidence = Fresnel loss. 60 is the plateau.
    d_hub_height: float = 60.0  # above plate top
    # Round 4 mm steel skewer, snug so the dog sits straight. Prints
    # slightly undersize; ream to fit. MEASURE yours - steel skewers vary
    # (and flat ones need a rectangular hole, not this round one).
    d_stick_hole_diameter: float = 4.1

    # ------------------------------------------------------------------
    # Clamp ring over the assembled quarter-post pedestal
    # ------------------------------------------------------------------
    d_ring_height: float = 12.0
    d_ring_wall: float = 2.2  # OD 19.5 clears the O20 dog shadow
    d_ring_clearance: float = 0.3  # radial slip fit, bore to post

    # ------------------------------------------------------------------
    # Hot dog (the load case)
    # ------------------------------------------------------------------
    # Rob's actual dogs, measured 2026-08-19.
    d_dog_length: float = 136.0
    d_dog_diameter: float = 20.0

    # ------------------------------------------------------------------
    # Mirror field layout
    # ------------------------------------------------------------------
    # A low-aimed beam from a far mirror skims flat across the mirror
    # field and can clip intermediate pad edges. Require every beam to
    # climb at least this much per mm of horizontal travel - it reserves
    # the low targets for inner mirrors, whose hop is short and clear.
    # 0.45, not the 0.3 this started at: the shallower floor handed far
    # mirrors the low, field-skimming aims that ray-traced worst.
    d_min_beam_slope: float = 0.45

    # Sites closer in than this are pointless: the dog shades r < 11 mm
    # itself, and closer beams graze the dog nearly parallel to its axis
    # (200 mm stripes, mostly missing, mostly reflecting off the skin).
    # 23.0 deliberately admits the four diagonal sites at r = 23.4 - the
    # corners of the central gap. Their light is grazing but real.
    d_min_site_radius: float = 23.0

    # ------------------------------------------------------------------
    # Optics / power model
    # ------------------------------------------------------------------
    d_solar_irradiance: float = 900.0  # W/m^2 direct-normal, clear day
    d_mirror_reflectivity: float = 0.85
    d_dog_absorptivity: float = 0.80
    d_sun_half_angle: float = 4.65e-3  # radians; the sun is not a point

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------
    @property
    def d_plate_half(self) -> float:
        return self.d_plate_size / 2.0

    @property
    def d_dog_radius(self) -> float:
        return self.d_dog_diameter / 2.0

    @property
    def d_hub_diameter(self) -> float:
        """Skewer hole + 5.2 mm walls (1.5x the original 3.45 - Rob
        wanted it beefier). Ø14.5 still sits entirely inside the seated
        dog's Ø22 shadow, so no ray can clip the pedestal edge."""
        return self.d_stick_hole_diameter + 2.0 * 5.2

    @property
    def d_dog_bottom_height(self) -> float:
        """The dog SITS on the hub - skewered through, resting on the hub
        top. Not a free parameter: the hub is the seat."""
        return self.d_hub_height

    @property
    def d_dog_top_height(self) -> float:
        return self.d_dog_bottom_height + self.d_dog_length

    @property
    def d_mirror_half(self) -> float:
        return self.d_mirror_size / 2.0

    @property
    def d_grid_pitch(self) -> float:
        """Site pitch: mirror edge to mirror edge is d_mirror_gap of air."""
        return self.d_mirror_size + self.d_mirror_gap

    @property
    def d_pad_reach(self) -> float:
        """Pad half-extent from site center (mirror + a little margin)."""
        return self.d_mirror_half + self.d_pad_extra

    @property
    def d_stack_height(self) -> float:
        """Glue + mirror above the pad surface."""
        return self.d_glue_gap + self.d_mirror_thickness

    @property
    def n_grid_cells(self) -> int:
        """Cells per side: largest EVEN count that fits the plate. Even is
        required - an odd grid puts sites on the fold planes."""
        d_usable = self.d_plate_half - self.d_plate_margin - self.d_pad_reach
        n = int(d_usable / self.d_grid_pitch - 0.5) * 2 + 2
        while (
            n / 2.0 - 0.5
        ) * self.d_grid_pitch + self.d_pad_reach > self.d_plate_half - self.d_plate_margin + 1e-6:
            n -= 2
        return n

    @property
    def d_sight_cutoff_deg(self) -> float:
        """Aim error at which a sight bore goes fully dark."""
        return math.degrees(
            math.atan(
                self.d_sight_hole_diameter / (self.d_sight_boss_height + self.d_base_thickness)
            )
        )

    @property
    def d_quadrant_size(self) -> float:
        """One printed plate = one quadrant of the assembled square."""
        return self.d_plate_size / 2.0

    @property
    def d_tab_center_y(self) -> float:
        """Tab centered along its quadrant's outer edge (Q1: +X edge)."""
        return self.d_quadrant_size / 2.0

    @property
    def d_sight_center_x(self) -> float:
        """Boss on the tab's centerline (no screw to flank)."""
        return self.d_plate_half + (self.d_sight_tab_length - self.d_sight_tab_root) / 2.0

    def pad_raise(self, d_r: float) -> float:
        """QUADRATIC stadium ramp (v1's was linear). The neighbor-edge
        deficit a beam must clear grows with pad tilt, which grows
        roughly linearly with radius - so the raise GRADIENT should too.
        A quadratic ramp puts its slope in the outer rings where the
        blocking actually happens, for the same 20 mm total."""
        d_r0 = self.d_min_site_radius
        d_r1 = (self.n_grid_cells / 2.0 - 0.5) * self.d_grid_pitch * math.sqrt(2.0)
        d_f = min(max((d_r - d_r0) / (d_r1 - d_r0), 0.0), 1.0)
        return self.d_perimeter_raise * d_f * d_f

    @property
    def d_ring_bore_radius(self) -> float:
        return self.d_hub_diameter / 2.0 + self.d_ring_clearance

    @property
    def d_ring_outer_radius(self) -> float:
        return self.d_ring_bore_radius + self.d_ring_wall

    def validate(self) -> list[str]:
        """See ``DiscoLensParameters.validate``. The printed footprint is
        the quadrant plus its tab, not the assembled plate, so the bed
        rules differ from v1's and are named separately."""
        out: list[str] = []
        if self.n_grid_cells < 4:
            out.append("grid_has_enough_cells: plate too small for a sensible mirror grid")
        if self.d_quadrant_size + self.d_sight_tab_length > 246.0:
            out.append("quadrant_fits_the_bed_x: quadrant + sight tab overruns the Prusa bed X")
        if self.d_quadrant_size > 208.0:
            out.append("quadrant_fits_the_bed_y: quadrant overruns the Prusa bed Y")
        if self.d_sight_hole_diameter >= self.d_sight_boss_diameter - 3.0:
            out.append("sight_bore_leaves_boss_wall: sight bore leaves too little boss wall")
        d_reach = (self.n_grid_cells / 2.0 - 0.5) * self.d_grid_pitch + self.d_pad_reach
        if d_reach + 1.0 > self.d_plate_half - self.d_rim_thickness:
            out.append("rim_clears_the_grid: perimeter rim collides with the mirror grid")
        if self.d_sight_tab_root <= self.d_plate_corner_radius:
            out.append(
                "sight_tab_clears_the_corner_arc: sight tab root "
                "must reach past the rounded corner arc"
            )
        if self.d_min_site_radius <= self.d_hub_diameter / 2.0 + 5.0:
            out.append("sites_clear_the_hub: site keep-out too tight against the hub")
        if self.d_stick_hole_diameter >= self.d_hub_diameter - 4.0:
            out.append("skewer_bore_leaves_hub_wall: stick hole leaves too little hub wall")
        # The ring must hide inside the seated dog's shadow like the hub.
        if 2.0 * self.d_ring_outer_radius >= self.d_dog_diameter:
            out.append("ring_hides_in_the_dog_shadow: clamp ring pokes out of the dog's shadow")
        return out

    def summary(self) -> str:
        return (
            f"disco-lens: {self.d_plate_size:.0f} mm assembled square "
            f"(4x {self.d_quadrant_size:.0f} mm quadrants), "
            f"{self.n_grid_cells}x{self.n_grid_cells} grid of "
            f"{self.d_mirror_size:.2f} mm mirrors, dog "
            f"Ø{self.d_dog_diameter:.0f}x{self.d_dog_length:.0f} at "
            f"z=[{self.d_dog_bottom_height:.0f}, {self.d_dog_top_height:.0f}]"
        )


def raise_if_invalid(p: DiscoLensParameters) -> None:
    """Raise if any rule is broken, reporting all of them at once.

    ``validate`` returns problems so a caller can look at them; this is
    the path that refuses to build. Every ``build_*_part`` entry point
    calls it, which is what keeps an invalid design from reaching a
    ``.sdm`` that would validate against the schema and be wrong.
    """
    problems = p.validate()
    if problems:
        raise ValueError(
            f"{type(p).__name__} has {len(problems)} broken "
            f"constraint(s):\n  " + "\n  ".join(problems)
        )
