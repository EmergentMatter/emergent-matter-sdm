"""Lay out the primitives-showcase board in the Blender GUI for framing.

A color-coded primitive grid with vertical category legends down the left,
plus a combined Lofts & Sweeps row at the bottom. Reads ``mega/manifest.json``
(and the STLs it points to) produced by ``build.py``.

    /Applications/Blender.app/Contents/MacOS/Blender --python layout.py

``bpy`` is imported lazily, inside :func:`build_layout`, not at module scope:
see that function's docstring for why.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

SHOWCASE_DIR = Path(__file__).resolve().parent

THEME = os.environ.get("SHOWCASE_THEME", "dark").lower()  # "dark" | "light"
LIGHT = THEME == "light"
TEXT_COLOR = (0.0, 0.0, 0.0) if LIGHT else (0.92, 0.92, 0.95)  # true black on white
S = 0.001
PXG, PZG = 0.072, 0.055  # grid pitch (tight)
PXB_ROW = 0.065  # single combined lofts+sweeps row pitch
GAP = 0.044
PROFILE_CAT = "profile_2d"
# Print page. Width / height of the portrait page the camera frames; 3:4 is
# the 18 x 24 in / 24 x 32 in poster family, and the closest standard shape to
# the board's own proportions. Override for another stock size, e.g. 2:3 for
# 24 x 36 in (POSTER_ASPECT=0.6667) or ISO A-series (POSTER_ASPECT=0.7071).
POSTER_ASPECT = float(os.environ.get("POSTER_ASPECT", "0.75"))
POSTER_MARGIN = 0.040  # world units of clear page on every side of the content
# Light poster drop shadows. The parts float LIGHT_BACKDROP_Y behind their own
# plane in front of a white backdrop; a soft sun from upper-left-front throws
# each one's shadow down and to the right onto it. The world is dimmed to
# LIGHT_WORLD so the shadow (world-lit only) reads as a grey of that value,
# and the sun is set so unshadowed backdrop comes out exactly 1.0 white.
LIGHT_WORLD = 0.68
LIGHT_BACKDROP_Y = 0.016  # just behind the deepest part (camera looks down +Y)
LIGHT_SUN_TILT = math.radians(16.0)  # off the view axis: sets the shadow offset
LIGHT_SUN_SOFTNESS = math.radians(3.0)  # angular size: sets the penumbra
# EEVEE sizes a light's shadow texels in world units and refuses to go below
# this floor. Its default (1 mm) is coarser than the parts, which are tens of
# millimetres across, and turns every shadow into a block the size of the
# caster; the floor has to sit well under the smallest feature. Metres.
LIGHT_SHADOW_TEXEL = 0.00002

# Concise category names for the vertical left-margin labels (color legend).
CATEGORY_LABELS = {
    "exact_3d": "EXACT 3-D",
    "tpms": "TPMS",
    "compliant": "COMPLIANT",
    "profile_2d": "2-D PROFILES",
}
BUILD_ROW_LABELS = {0: "LOFTS", 1: "SWEEPS"}

# Vivid per-family palette for the white poster. The catalog's saturated
# primaries read as washed-out pastel under the flat white-world exposure, so
# the light theme swaps in these hand-picked, richer hues. 2-D profiles are
# left on the catalog blue (they already read well as thin outlines, handled
# by the desaturating fallback in material_for()).
LIGHT_PALETTE = {
    "exact_3d": (0.95, 0.42, 0.10),  # warm orange
    "tpms": (0.05, 0.62, 0.55),  # teal / emerald
    "compliant": (0.90, 0.20, 0.38),  # rose / raspberry
    "construction": (0.52, 0.26, 0.88),  # violet (lofts & sweeps)
}


def row_z(cell: dict[str, Any], build_z: float, prof_top_z: float, prof_rows: list[int]) -> float:
    """A grid cell's vertical placement: the profile block sits below the
    combined lofts/sweeps band, everything else stacks by its own row."""
    if cell["category"] == PROFILE_CAT:
        return prof_top_z - (cell["row"] - prof_rows[0]) * PZG
    return -cell["row"] * PZG


def build_layout() -> None:
    """Build the showcase board in the current Blender scene.

    ``bpy`` only exists inside Blender's embedded Python, so it is imported
    here rather than at module scope: that keeps a plain
    ``import sdm_showcase.layout`` (and this module's own pure helpers, like
    :func:`row_z`) usable from a normal Python environment for testing, and
    it is what lets ``verify-wheel``'s plain ``import sdm_showcase`` succeed
    outside Blender.
    """
    try:
        import bpy
    except ImportError as exc:
        raise RuntimeError(
            "layout.py must run inside Blender's Python (bpy is a "
            "Blender-embedded module, not pip-installable). Run it as "
            "`blender --python layout.py`."
        ) from exc

    manifest = SHOWCASE_DIR / "mega" / "manifest.json"
    cells = json.loads(manifest.read_text())
    grid = [c for c in cells if c["section"] == "grid"]
    build = [c for c in cells if c["section"] == "build"]

    # -- vertical layout -------------------------------------------------
    # Stacking order (top -> bottom): the non-profile grid families, then
    # the combined Lofts & Sweeps band, then 2-D profiles as the bottom
    # block.
    max_top_row = max(c["row"] for c in grid if c["category"] != PROFILE_CAT)
    prof_rows = sorted({c["row"] for c in grid if c["category"] == PROFILE_CAT})
    build_z = -max_top_row * PZG - GAP - 0.022  # band below last top family
    prof_top_z = build_z - GAP - 0.022  # profiles below the band
    prof_bottom_z = prof_top_z - (prof_rows[-1] - prof_rows[0]) * PZG

    def z_of(cell: dict[str, Any]) -> float:
        return row_z(cell, build_z, prof_top_z, prof_rows)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    mat_cache: dict[str, Any] = {}

    def material_for(key: str, color: tuple[float, float, float]) -> Any:
        if key not in mat_cache:
            if LIGHT:
                if key in LIGHT_PALETTE:
                    color = LIGHT_PALETTE[key]
                else:
                    # fallback (2-D profiles): gently desaturate the catalog hue
                    g = sum(color) / 3.0
                    r, gr, bl = color
                    color = (g + (r - g) * 0.42, g + (gr - g) * 0.42, g + (bl - g) * 0.42)
            m = bpy.data.materials.new(key)
            m.use_nodes = True
            b = m.node_tree.nodes["Principled BSDF"]
            b.inputs["Base Color"].default_value = (*color, 1.0)
            b.inputs["Roughness"].default_value = 0.38
            m.diffuse_color = (*color, 1.0)
            mat_cache[key] = m
        return mat_cache[key]

    def flatten_text(bsdf: Any) -> None:
        """Light poster: kill specular + max roughness so black type reads as
        true black under the bright white world instead of catching a grey
        sheen."""
        if not LIGHT:
            return
        for name in ("Specular IOR Level", "Specular"):
            if name in bsdf.inputs:
                bsdf.inputs[name].default_value = 0.0
        bsdf.inputs["Roughness"].default_value = 1.0

    lbl_mat = bpy.data.materials.new("label")
    lbl_mat.use_nodes = True
    lbl_bsdf = lbl_mat.node_tree.nodes["Principled BSDF"]
    lbl_bsdf.inputs["Base Color"].default_value = (*TEXT_COLOR, 1.0)
    flatten_text(lbl_bsdf)

    def add_label(text: str, x: float, z: float, size: float = 0.0054) -> None:
        txt = bpy.data.curves.new(text, type="FONT")
        txt.body = text
        txt.align_x = "CENTER"
        ob = bpy.data.objects.new(text + "_lbl", txt)
        ob.location = (x, -0.018, z)
        ob.rotation_euler = (math.pi / 2, 0, 0)
        ob.scale = (size, size, size)
        ob.data.materials.append(lbl_mat)
        ob.visible_shadow = not LIGHT  # white poster: type casts no drop shadow
        bpy.context.collection.objects.link(ob)

    def add_row_label(
        text: str, x: float, z: float, color: tuple[float, float, float], size: float = 0.0072
    ) -> None:
        """A vivid, vertical (bottom-to-top) category label down the left
        margin, in the row's own hue, so it doubles as a color legend."""
        txt = bpy.data.curves.new(text, type="FONT")
        txt.body = text
        txt.align_x = "CENTER"
        txt.align_y = "CENTER"
        ob = bpy.data.objects.new(text + "_rowlbl", txt)
        ob.location = (x, -0.018, z)
        ob.rotation_mode = "ZYX"
        ob.rotation_euler = (math.pi / 2, 0, math.pi / 2)  # face -Y, read bottom->top
        ob.scale = (size, size, size)
        col = TEXT_COLOR if LIGHT else color  # white poster: keep all type black
        m = bpy.data.materials.new(text + "_rowmat")
        m.use_nodes = True
        b = m.node_tree.nodes["Principled BSDF"]
        b.inputs["Base Color"].default_value = (*col, 1.0)
        b.inputs["Roughness"].default_value = 0.4
        flatten_text(b)
        m.diffuse_color = (*col, 1.0)
        ob.data.materials.append(m)
        ob.visible_shadow = not LIGHT
        bpy.context.collection.objects.link(ob)

    def place(cell: dict[str, Any], x: float, z: float) -> None:
        bpy.ops.wm.stl_import(filepath=str(SHOWCASE_DIR / cell["stl"]))
        obj = bpy.context.selected_objects[0]
        obj.name = cell["name"]
        obj.scale = (S, S, S)
        bpy.ops.object.transform_apply(scale=True)
        bpy.ops.object.shade_smooth()
        obj.location = (x, 0.0, z)
        obj.data.materials.append(material_for(cell["category"], tuple(cell["color"])))
        # White poster: the 2-D profiles are thin outlines drawn on the page,
        # not objects standing in front of it, so they cast no drop shadow.
        if LIGHT and cell["category"] == PROFILE_CAT:
            obj.visible_shadow = False

    # -- primitive grid ----------------------------------------------------
    # Every row is centred on its own cell count, so a family that does not
    # divide evenly into its column count ends in a shorter centred row (and
    # families with different column counts share one centre line).
    row_len: dict[tuple[str, int], int] = {}
    for c in grid:
        key = (c["category"], c["row"])
        row_len[key] = row_len.get(key, 0) + 1
    widest = max(row_len.values())
    widest_x = 0.0
    for c in grid:
        x = (c["col"] - (row_len[(c["category"], c["row"])] - 1) / 2.0) * PXG
        z = z_of(c)
        place(c, x, z)
        add_label(c["name"], x, z - 0.019)
        widest_x = max(widest_x, x)

    # -- lofts & sweeps: one combined row (above the 2-D profile block) ----
    build_right = 0.0
    for j, c in enumerate(build):
        x = (j - (len(build) - 1) / 2.0) * PXB_ROW
        place(c, x, build_z)
        add_label(c["name"], x, build_z - 0.030)
        build_right = max(build_right, x)

    # -- per-category vertical row labels (left margin; double as color legend)
    label_x = -(widest - 1) / 2.0 * PXG - 0.050
    cats: list[str] = []
    for c in grid:
        if c["category"] not in cats:
            cats.append(c["category"])
    for cat in cats:
        zs = sorted({z_of(c) for c in grid if c["category"] == cat})
        color = next(tuple(c["color"]) for c in grid if c["category"] == cat)
        zc = (zs[0] + zs[-1]) / 2.0
        add_row_label(CATEGORY_LABELS.get(cat, cat.upper()), label_x, zc, color)

    build_color = next((tuple(c["color"]) for c in build), (0.62, 0.12, 0.95))
    add_row_label("LOFTS & SWEEPS", label_x, build_z, build_color)

    # -- title, footer ---------------------------------------------------------
    # Poster masthead: the title sits just above the grid, the URL is the
    # footer below it, so the type frames the board instead of sitting in a
    # lump at the top. Extents feed the camera framing below.
    title_z = 0.052
    footer_z = prof_bottom_z - 0.052
    add_label("SOFTWARE DEFINED MATTER PRIMITIVES", 0.0, title_z, size=0.0094)
    add_label("www.emergentmatter.com", 0.0, footer_z, size=0.0066)

    # Content box (world units): top of the title to the bottom of the footer,
    # left legend to the right-most cell edge.
    content_top = title_z + 0.010
    content_bottom = footer_z - 0.006
    content_left = label_x - 0.010
    content_right = max(widest_x, build_right) + 0.036
    scene_mid_z = (content_top + content_bottom) / 2.0
    scene_mid_x = (content_left + content_right) / 2.0

    # -- camera (orthographic, perfectly axis-aligned: dead-on -Y, zero roll)
    # Both themes share the same flat orthographic framing: no perspective.
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam_data.type = "ORTHO"
    # Frame the content box inside a POSTER_ASPECT portrait page with the
    # same margin on every side. Blender's ortho_scale is the larger frame
    # dimension, which for a portrait page is the height. render.py derives
    # its pixel size from the same aspect, so the page and the pixels agree.
    content_w = content_right - content_left
    content_h = content_top - content_bottom
    page_h = max(content_h + 2 * POSTER_MARGIN, (content_w + 2 * POSTER_MARGIN) / POSTER_ASPECT)
    cam_data.ortho_scale = page_h
    cam.location = (scene_mid_x, -2.0, scene_mid_z)
    cam.rotation_euler = (math.pi / 2, 0.0, 0.0)  # look straight down +Y, up = +Z
    print(
        f"Page: aspect {POSTER_ASPECT:.3f}, world {page_h * POSTER_ASPECT:.3f} x {page_h:.3f}, "
        f"content {content_w:.3f} x {content_h:.3f}"
    )

    # -- lights ------------------------------------------------------------
    # Dark theme only: on the white poster the sun below is the single lamp,
    # because any other lamp also lights the backdrop and lifts the shadow
    # back to white (the world fill already supplies the paper's base level).
    area_lights = [
        ("Key", 260, 0.9, (0.8, -1.2, scene_mid_z + 0.9)),
        ("Fill", 110, 1.4, (-1.0, -0.9, scene_mid_z + 0.4)),
        ("Rim", 170, 0.7, (-0.4, 1.2, scene_mid_z + 0.8)),
    ]
    for name, energy, size, loc in [] if LIGHT else area_lights:
        light = bpy.data.lights.new(name, type="AREA")
        light.energy = energy
        light.size = size
        ol = bpy.data.objects.new(name, light)
        ol.location = loc
        bpy.context.collection.objects.link(ol)

    if LIGHT:
        # Backdrop the shadows land on: white, matte, facing the camera. Its
        # unshadowed value under the Standard transform is world + sun/pi,
        # so the sun is sized to bring that to exactly 1.0; in shadow only
        # the world remains and the backdrop drops to LIGHT_WORLD grey.
        bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, LIGHT_BACKDROP_Y, scene_mid_z))
        backdrop = bpy.context.active_object
        backdrop.name = "Backdrop"
        backdrop.rotation_euler = (math.pi / 2, 0.0, 0.0)  # normal toward -Y (the camera)
        paper = bpy.data.materials.new("paper")
        paper.use_nodes = True
        pb = paper.node_tree.nodes["Principled BSDF"]
        pb.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        pb.inputs["Roughness"].default_value = 1.0
        flatten_text(pb)  # no specular sheen on the paper either
        backdrop.data.materials.append(paper)

        # Light travels into the backdrop (+Y), a little downward and to the
        # right, so shadows fall down-right of each part. A sun points along
        # its local -Z; track that axis onto the travel direction.
        from mathutils import Vector  # type: ignore[import-not-found]  # Blender-embedded

        travel = Vector((math.sin(LIGHT_SUN_TILT) * 0.7, 1.0, -math.sin(LIGHT_SUN_TILT) * 0.7))
        travel.normalize()
        sun_data = bpy.data.lights.new("Sun", type="SUN")
        sun_data.angle = LIGHT_SUN_SOFTNESS
        sun_data.shadow_maximum_resolution = LIGHT_SHADOW_TEXEL
        sun_data.shadow_filter_radius = 0.5  # default 1.0 smears the penumbra into a halo
        bpy.context.scene.eevee.shadow_ray_count = 4  # smoother penumbra at print scale
        bpy.context.scene.eevee.shadow_step_count = 8
        # White Lambertian paper under irradiance E reads E/pi; the backdrop
        # sees the sun at incidence travel.y, so this lands unshadowed paper
        # on exactly 1.0 above the LIGHT_WORLD fill.
        sun_data.energy = (1.0 - LIGHT_WORLD) * math.pi / travel.y
        sun = bpy.data.objects.new("Sun", sun_data)
        sun.rotation_euler = travel.to_track_quat("-Z", "Y").to_euler()
        bpy.context.collection.objects.link(sun)

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bgn = world.node_tree.nodes["Background"]
    if LIGHT:
        # Uniform white world as fill. Below full strength on purpose: the
        # backdrop reaches 1.0 only with the sun on it, so wherever a part
        # blocks the sun the paper shows this value, which is the drop
        # shadow. Parts still expose near their albedo.
        bgn.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        bgn.inputs["Strength"].default_value = LIGHT_WORLD
    else:
        bgn.inputs["Color"].default_value = (0.03, 0.035, 0.045, 1.0)
        bgn.inputs["Strength"].default_value = 0.55
    bpy.context.scene.world = world

    if LIGHT:
        # Default AgX tone-maps a 1.0 world down to grey; Standard keeps the
        # background pure white and the black type genuinely black.
        bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.render.engine = "BLENDER_EEVEE"

    def setup_viewport() -> None:
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type != "VIEW_3D":
                    continue
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "MATERIAL"
                        space.shading.color_type = "MATERIAL"
                        space.region_3d.view_perspective = "CAMERA"

    bpy.app.timers.register(setup_viewport, first_interval=0.4)
    print(f"Board ready: {len(grid)} primitives + {len(build)} constructions.")


if __name__ == "__main__":
    build_layout()
