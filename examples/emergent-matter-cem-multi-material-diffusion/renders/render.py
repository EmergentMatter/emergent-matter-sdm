"""Pictures of the diffusion bar. Nothing here is needed to build the part.

This whole folder is presentation. `../build.py` and `../cem.toml` are the CEM;
they turn parameters into geometry and know nothing about cameras. If you are
here to understand the part, you are one directory too deep.

Two modes, chosen by how Blender is started rather than by a flag:

    # headless -> writes hero.png next to this file
    blender --background --python renders/render.py

    # interactive -> sets the viewport to the hero camera and hands you the
    # scene to orbit
    blender --python renders/render.py

Reads `../manifest.json` and the STLs it names, so run `../build.py` first.

Environment:
    SDM_VIEW=cut|solid   cutaway (default) or the whole monolithic bar
    SDM_HERO_CAM=persp|ortho   projection; see add_camera()
    SDM_HERO_OUT=path    where the headless render lands

TRUE ISOMETRIC is the ANGLE, not the projection: Euler (54.736, 0, 45) degrees,
where 54.736 is atan(1/sqrt(2)) -- the elevation that projects the three axes
120 degrees apart with equal foreshortening. Both camera modes sit there and
differ only in projection, so the two shots are directly comparable.
"""

import json
import math
import os
from pathlib import Path

import bpy

HERE = Path(__file__).resolve().parent
PART = HERE.parent  # the CEM lives one up
MANIFEST = json.loads((PART / "manifest.json").read_text())

MODE = os.environ.get("SDM_VIEW", "cut").lower()
REGIONS = MANIFEST[MODE]
OUT = os.environ.get("SDM_HERO_OUT", str(HERE / "hero.png"))

S = 0.01  # mm -> Blender units; the 200 mm bar is 2.0 long
RES_X, RES_Y = 2000, 1500  # 4:3
SAMPLES = 128

#: "persp" or "ortho". Persp gives the bar depth and a receding long axis, so
#: it reads as an object. Ortho is the technical read: equal foreshortening
#: everywhere, so a particle's size is PURELY its composition and never its
#: distance from the camera.
CAM_MODE = os.environ.get("SDM_HERO_CAM", "persp").lower()

#: Fraction of the frame the subject fills. Under 1.0 leaves margin all round.
FILL = 0.80

#: Focal length for the perspective shot, mm on a 36 mm sensor. Long enough
#: that the near end does not balloon: at 200 mm the bar is long relative to
#: any sane camera distance, and a wide lens turns "isometric-ish" into a
#: dramatic diagonal that fights the gradient for attention.
LENS_MM = 85.0


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for coll in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.worlds,
    ):
        for block in list(coll):
            coll.remove(block)


def material(name, color):
    """Slightly waxy dielectric. Low roughness reads as plastic and blows out
    the highlight at 2000 px; 0.42 keeps the spheres legible where they crowd
    together at the weld."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = 0.42
    if "Metallic" in b.inputs:
        b.inputs["Metallic"].default_value = 0.0
    for key, val in (("Specular IOR Level", 0.35), ("Coat Weight", 0.15)):
        if key in b.inputs:
            b.inputs[key].default_value = val
    m.diffuse_color = (*color, 1.0)
    return m


def import_region(entry):
    bpy.ops.wm.stl_import(filepath=str(PART / entry["stl"]))
    obj = bpy.context.selected_objects[0]
    obj.name = entry["name"]
    obj.scale = (S, S, S)
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.shade_smooth()
    obj.data.materials.append(material(entry["name"], tuple(entry["color"])))
    return obj


def add_lights():
    """Three-point rig plus a soft underfill, sized in scene units.

    The subject is 2.0 x 0.2 x 0.2 -- a sliver. Small lamps give it a row of
    hard specular dots that read as noise along the length, so the key and fill
    are DELIBERATELY LARGER THAN THE BAR: a broad source wraps a cylinder and
    leaves one long soft highlight instead of two hundred small ones.

    Watts. An earlier pass ran these at 1400/420/900 and clipped the top face
    to pure white -- on a 0.2-unit-thick bar the key is nearly normal to that
    face, so it takes the full beam. These keep the brightest highlight under
    1.0 with the "Standard" view transform, which is what preserves the red and
    blue as distinguishable rather than as two pastels.
    """
    for name, energy, size, loc in (
        ("Key", 260, 4.0, (2.4, -3.2, 2.0)),
        ("Fill", 90, 6.0, (-3.0, -2.4, 0.9)),
        ("Rim", 190, 2.5, (-1.6, 2.8, 2.2)),
        ("Under", 35, 4.0, (0.0, -1.4, -2.2)),
    ):
        light = bpy.data.lights.new(name, type="AREA")
        light.energy = energy
        light.size = size
        ob = bpy.data.objects.new(name, light)
        ob.location = loc
        target = bpy.data.objects.new(f"{name}_target", None)
        target.location = (0.0, 0.0, 0.0)
        bpy.context.collection.objects.link(target)
        bpy.context.collection.objects.link(ob)
        con = ob.constraints.new(type="TRACK_TO")
        con.target = target
        con.track_axis = "TRACK_NEGATIVE_Z"
        con.up_axis = "UP_Y"

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.021, 0.024, 0.031, 1.0)
    bg.inputs["Strength"].default_value = 0.30
    bpy.context.scene.world = world


def _bounds_centre():
    """Centre of the scene's mesh bounding box, in world space.

    The starting aim only -- `_frame_to_projection` corrects it in screen
    space afterwards, which is where the answer actually has to be right.
    """
    import mathutils

    lo = [None] * 3
    hi = [None] * 3
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH":
            continue
        for corner in ob.bound_box:
            w = ob.matrix_world @ mathutils.Vector(corner)
            for i in range(3):
                lo[i] = w[i] if lo[i] is None else min(lo[i], w[i])
                hi[i] = w[i] if hi[i] is None else max(hi[i], w[i])
    if lo[0] is None:
        return (0.0, 0.0, 0.0)
    return tuple((lo[i] + hi[i]) * 0.5 for i in range(3))


def _projected_width(v):
    """Extent of the scene across the camera's horizontal axis, in units.

    Measured from the real mesh bounds rather than assuming a 2.0-long bar: the
    cutaway is shorter than the solid, and red and blue reach different
    distances once `balance` is off zero, so a hard-coded number mis-frames
    every variant but the default.
    """
    import mathutils

    up = mathutils.Vector((0.0, 0.0, 1.0))
    fwd = mathutils.Vector(v).normalized()
    right = fwd.cross(up).normalized()
    lo = hi = None
    for ob in bpy.context.scene.objects:
        if ob.type != "MESH":
            continue
        for corner in ob.bound_box:
            d = (ob.matrix_world @ mathutils.Vector(corner)).dot(right)
            lo = d if lo is None else min(lo, d)
            hi = d if hi is None else max(hi, d)
    return max(hi - lo, 1e-6) if lo is not None else 2.0


def add_camera():
    """Camera at the isometric angle, framed from the SUBJECT'S PROJECTION."""
    d_el = math.radians(54.736)
    d_az = math.radians(45.0)
    d_dip = math.radians(35.264)  # 90 - 54.736
    v = (math.sin(d_az) * math.cos(d_dip), -math.cos(d_az) * math.cos(d_dip), math.sin(d_dip))

    c = _bounds_centre()
    d_wide = _projected_width(v)

    cam_data = bpy.data.cameras.new("Camera")
    if CAM_MODE == "ortho":
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = d_wide / FILL
        d_dist = 8.0
    else:
        cam_data.type = "PERSP"
        cam_data.lens = LENS_MM
        cam_data.sensor_fit = "HORIZONTAL"
        d_dist = (d_wide / FILL) * cam_data.lens / cam_data.sensor_width

    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = (c[0] + v[0] * d_dist, c[1] + v[1] * d_dist, c[2] + v[2] * d_dist)
    cam.rotation_euler = (d_el, 0.0, d_az)
    cam_data.clip_start = max(0.05, d_dist - 6.0)
    cam_data.clip_end = d_dist + 6.0
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    _frame_to_projection(cam, d_dist)
    return cam


def _frame_to_projection(cam, d_dist, n_passes=4):
    """Centre and size the subject using its PROJECTED bounds.

    Aiming at the bounding box's centre is not the same as centring what the
    camera sees: under perspective the near half of a long diagonal subtends
    more than the far half, so the silhouette lands off-centre even when the
    aim is exactly right. Measuring the projected box and correcting for it is
    the only version that puts even margin on all four sides, which is the
    whole point of the 4:3 crop.

    Iterated because each correction changes the projection slightly; it
    converges in two or three passes.
    """
    import mathutils
    from bpy_extras.object_utils import world_to_camera_view

    scene = bpy.context.scene
    d_aspect = scene.render.resolution_y / scene.render.resolution_x

    for _ in range(n_passes):
        bpy.context.view_layer.update()
        xs, ys = [], []
        for ob in scene.objects:
            if ob.type != "MESH":
                continue
            for corner in ob.bound_box:
                w = ob.matrix_world @ mathutils.Vector(corner)
                ndc = world_to_camera_view(scene, cam, w)
                xs.append(ndc.x)
                ys.append(ndc.y)
        if not xs:
            return
        d_cx, d_cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        d_fx, d_fy = max(xs) - min(xs), max(ys) - min(ys)

        if cam.data.type == "ORTHO":
            d_ext_x = cam.data.ortho_scale
        else:
            d_ext_x = (cam.data.sensor_width / cam.data.lens) * d_dist
        d_ext_y = d_ext_x * d_aspect

        m = cam.matrix_world
        right = mathutils.Vector((m[0][0], m[1][0], m[2][0]))
        up = mathutils.Vector((m[0][1], m[1][1], m[2][1]))
        # PLUS, not minus: moving the camera right slides the IMAGE left, so to
        # bring a subject sitting at cx back to 0.5 the camera travels the same
        # way the subject is off. Backwards, this compounds every pass and
        # walks the subject out of frame entirely.
        cam.location += right * ((d_cx - 0.5) * d_ext_x)
        cam.location += up * ((d_cy - 0.5) * d_ext_y)

        # Fit whichever axis binds, so the margin is even rather than merely
        # wide. Both fractions are already normalised to their OWN axis --
        # world_to_camera_view divides x by the frame width and y by its
        # height -- so they compare directly and dividing by the aspect
        # double-counts it.
        d_k = max(d_fx, d_fy) / FILL
        if cam.data.type == "ORTHO":
            cam.data.ortho_scale *= d_k
        else:
            fwd = mathutils.Vector((m[0][2], m[1][2], m[2][2]))  # +Z is BACK
            d_new = d_dist * d_k
            cam.location += fwd * (d_new - d_dist)
            d_dist = d_new
            cam.data.clip_start = max(0.05, d_dist - 6.0)
            cam.data.clip_end = d_dist + 6.0
        if abs(d_cx - 0.5) < 2e-4 and abs(d_cy - 0.5) < 2e-4 and abs(d_k - 1.0) < 2e-3:
            break


def pick_engine(scene):
    """Blender renamed EEVEE between 4.1 and 4.2; support both rather than
    pinning a version this example does not otherwise care about."""
    available = [
        e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    ]
    for name in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        if name in available:
            scene.render.engine = name
            return name
    return scene.render.engine


def main():
    clear_scene()
    for entry in REGIONS:
        import_region(entry)
    add_lights()

    scene = bpy.context.scene
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    add_camera()

    engine = pick_engine(scene)
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.compression = 100
    if hasattr(scene, "eevee"):
        for attr, val in (
            ("taa_render_samples", SAMPLES),
            ("use_gtao", True),
            ("use_bloom", False),
        ):
            if hasattr(scene.eevee, attr):
                setattr(scene.eevee, attr, val)
    if hasattr(scene, "cycles"):
        scene.cycles.samples = SAMPLES
    # Filmic/AgX crushes saturated red and blue toward brown. Standard keeps
    # the two materials distinguishable, which is the point of the image.
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
    except Exception:
        pass

    if bpy.app.background:
        scene.render.filepath = OUT
        print(f"[render] engine={engine} {RES_X}x{RES_Y} cam={CAM_MODE} mode={MODE}")
        bpy.ops.render.render(write_still=True)
        print("WROTE", OUT)
        return

    def _setup_viewport():
        # Returning nothing (None) tells Blender's timer not to re-run this.
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type != "VIEW_3D":
                    continue
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.shading.type = "MATERIAL"
                        space.region_3d.view_perspective = "CAMERA"

    bpy.app.timers.register(_setup_viewport, first_interval=0.4)
    print(f"[render] {MODE} view, {CAM_MODE} camera — orbit with MMB, scroll to zoom.")


main()
