"""The picture at the top of the README. Nothing here is needed to build the part.

This folder is presentation. The package, ``cem.toml`` and ``scripts/`` are
the CEM; they turn parameters into geometry and know nothing about cameras.

Two steps, because Blender's Python cannot import the package, and a third
to fit the hub's 500 KB image budget:

    uv run python renders/scene.py                     # mirror poses -> scene.json
    blender --background --python renders/render.py    # -> renders/hero.png
    magick renders/hero.png -alpha off -colors 256 -dither FloydSteinberg \
        PNG8:renders/hero.png                          # ~1 MB -> ~130 KB

The last step is a 256-colour palette. The field of 1284 small tiles is all
high-frequency detail, so full-colour PNG stays near 1 MB even at this size,
while the palette version is visibly the same picture.

Run it without ``--background`` to get the same scene on the hero camera in a
viewport you can orbit. Needs the STLs, so run ``scripts/export_stl.py`` first.

Environment:
    SDM_HERO_OUT=path    where the headless render lands
    SDM_HERO_SAMPLES=n   Cycles samples (default 256)
    SDM_HERO_PCT=n       resolution percentage, for quick drafts (default 100)

Cycles, not EEVEE: the picture is 1284 flat mirrors, and a screen-space
reflection shows each one as whatever happens to be on screen, which is
mostly other mirrors. The world is a sky above the horizon and a dark studio
below it, so the camera sees a dark backdrop and every tile shows a patch of
sky: the tiles read as glass because they are the only bright thing that
is not lit directly.
"""

import json
import math
import os
from pathlib import Path

import bpy
import mathutils

HERE = Path(__file__).resolve().parent
SCENE = json.loads((HERE / "scene.json").read_text())
OUT = os.environ.get("SDM_HERO_OUT", str(HERE / "hero.png"))
SAMPLES = int(os.environ.get("SDM_HERO_SAMPLES", "256"))
PCT = int(os.environ.get("SDM_HERO_PCT", "100"))

S = 0.01  # mm -> Blender units; the 410 mm plate is 4.1 across
RES_X, RES_Y = 1280, 960  # 4:3, the size of the other examples' heroes

#: Fraction of the frame the subject fills. Under 1.0 leaves margin all round.
FILL = 0.86

#: Camera direction: azimuth off the +x seam, and elevation above the plate.
#: 28 degrees is low enough that the dog stands up out of the field and the
#: tilt of the outer pads shows, high enough to see all four quadrants.
CAM_AZ_DEG = 38.0
CAM_EL_DEG = 32.0
LENS_MM = 70.0


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


def material(name, color, roughness, metallic=0.0, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = roughness
    b.inputs["Metallic"].default_value = metallic
    if "Coat Weight" in b.inputs:
        b.inputs["Coat Weight"].default_value = coat
    m.diffuse_color = (*color, 1.0)
    return m


def link(ob):
    bpy.context.collection.objects.link(ob)
    return ob


def import_stl(path, name, mat):
    bpy.ops.wm.stl_import(filepath=path)
    ob = bpy.context.selected_objects[0]
    ob.name = name
    ob.scale = (S, S, S)
    bpy.ops.object.transform_apply(scale=True)
    ob.data.materials.append(mat)
    return ob


def add_plate():
    """Four instances of ONE quadrant mesh, as the cooker is really built:
    one file printed four times. Linked duplicates share the 2.7 M-triangle
    mesh instead of holding four copies of it."""
    petg = material("PETG", (0.80, 0.78, 0.73), 0.55)
    quad = import_stl(SCENE["quadrant_stl"], "quadrant_0", petg)
    for n in range(1, 4):
        dup = link(bpy.data.objects.new(f"quadrant_{n}", quad.data))
        dup.rotation_euler = (0.0, 0.0, n * math.pi / 2.0)
    ring = import_stl(SCENE["ring_stl"], "ring", petg)
    ring.location.z = SCENE["ring_z"] * S


def add_mirrors():
    """Every tile as a box in ONE mesh: 1284 objects cost Blender more than
    the render does."""
    d_h = 0.5 * SCENE["mirror_size"]
    d_t = 0.5 * SCENE["mirror_thickness"]
    l_corner = [
        (sx * d_h, sy * d_h, sz * d_t) for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)
    ]
    l_face = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    l_verts, l_faces = [], []
    for tile in SCENE["mirrors"]:
        r = mathutils.Matrix(tile["rotation"])
        c = mathutils.Vector(tile["center"])
        n_base = len(l_verts)
        l_verts += [tuple((c + r @ mathutils.Vector(v)) * S) for v in l_corner]
        l_faces += [tuple(n_base + i for i in f) for f in l_face]
    mesh = bpy.data.meshes.new("mirrors")
    mesh.from_pydata(l_verts, [], l_faces)
    mesh.materials.append(material("mirror", (0.96, 0.96, 0.97), 0.015, 1.0))
    link(bpy.data.objects.new("mirrors", mesh))


def add_dog():
    dog = SCENE["dog"]
    d_r = dog["radius"] * S
    d_z0, d_z1 = dog["bottom"] * S + d_r, dog["top"] * S - d_r
    frank = material("frank", (0.50, 0.14, 0.06), 0.42, coat=0.08)
    # Less specular than the default: with a field of bright tiles all
    # round it, a normal sheen turns the lower half of the dog white.
    frank.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0.2
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=96, radius=d_r, depth=d_z1 - d_z0, location=(0, 0, 0.5 * (d_z0 + d_z1))
    )
    body = bpy.context.object
    for d_z in (d_z0, d_z1):
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=96, ring_count=48, radius=d_r, location=(0, 0, d_z)
        )
    bpy.ops.object.select_all(action="DESELECT")
    for ob in bpy.context.scene.objects:
        if ob.type == "MESH" and ob.name.startswith(("Cylinder", "Sphere")):
            ob.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.join()
    body.name = "dog"
    bpy.ops.object.shade_smooth()
    body.data.materials.append(frank)

    stick = SCENE["skewer"]
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=48,
        radius=stick["radius"] * S,
        depth=(stick["top"] - stick["bottom"]) * S,
        location=(0, 0, 0.5 * (stick["top"] + stick["bottom"]) * S),
    )
    bpy.context.object.name = "skewer"
    bpy.ops.object.shade_smooth()
    bpy.context.object.data.materials.append(material("steel", (0.78, 0.78, 0.80), 0.22, 1.0))


def add_lights():
    """The sun, nearly overhead, as it is when the cooker is aimed; and a
    world that is sky above the horizon, dark studio below."""
    sun = bpy.data.lights.new("Sun", type="SUN")
    sun.energy = 4.5
    sun.angle = math.radians(1.0)
    ob = link(bpy.data.objects.new("Sun", sun))
    ob.rotation_euler = (math.radians(14.0), math.radians(-8.0), 0.0)

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    nt = world.node_tree
    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    stops = ramp.color_ramp.elements
    stops[0].position, stops[0].color = 0.48, (0.020, 0.023, 0.030, 1.0)
    stops[1].position, stops[1].color = 1.00, (0.035, 0.10, 0.36, 1.0)
    horizon = stops.new(0.53)
    horizon.color = (0.75, 0.82, 0.92, 1.0)
    remap = nt.nodes.new("ShaderNodeMapRange")  # z in [-1, 1] -> [0, 1]
    remap.inputs["From Min"].default_value = -1.0
    nt.links.new(coord.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], remap.inputs["Value"])
    nt.links.new(remap.outputs["Result"], ramp.inputs["Fac"])
    bg = nt.nodes["Background"]
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0
    bpy.context.scene.world = world


def _mesh_corners():
    for ob in bpy.context.scene.objects:
        if ob.type == "MESH":
            for corner in ob.bound_box:
                yield ob.matrix_world @ mathutils.Vector(corner)


def add_camera():
    """Aim at the bounding box, then correct in SCREEN space: under
    perspective the near half of the plate subtends more than the far half,
    so aiming at the centre is not the same as centring the picture."""
    from bpy_extras.object_utils import world_to_camera_view

    scene = bpy.context.scene
    d_az, d_el = math.radians(CAM_AZ_DEG), math.radians(CAM_EL_DEG)
    v = mathutils.Vector(
        (math.cos(d_el) * math.cos(d_az), math.cos(d_el) * math.sin(d_az), math.sin(d_el))
    )
    l_pts = list(_mesh_corners())
    c = sum(l_pts, mathutils.Vector()) / len(l_pts)

    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = LENS_MM
    cam_data.sensor_fit = "HORIZONTAL"
    d_dist = 12.0
    cam = link(bpy.data.objects.new("Camera", cam_data))
    cam.location = c + v * d_dist
    cam.rotation_euler = (-v).to_track_quat("-Z", "Y").to_euler()
    scene.camera = cam

    for _ in range(6):
        bpy.context.view_layer.update()
        l_ndc = [world_to_camera_view(scene, cam, w) for w in l_pts]
        xs, ys = [p.x for p in l_ndc], [p.y for p in l_ndc]
        d_cx, d_cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
        d_k = max(max(xs) - min(xs), max(ys) - min(ys)) / FILL
        d_ext_x = (cam_data.sensor_width / cam_data.lens) * d_dist
        d_ext_y = d_ext_x * RES_Y / RES_X
        m = cam.matrix_world
        right = mathutils.Vector((m[0][0], m[1][0], m[2][0]))
        up = mathutils.Vector((m[0][1], m[1][1], m[2][1]))
        back = mathutils.Vector((m[0][2], m[1][2], m[2][2]))
        cam.location += right * ((d_cx - 0.5) * d_ext_x)
        cam.location += up * ((d_cy - 0.5) * d_ext_y)
        cam.location += back * (d_dist * d_k - d_dist)
        d_dist *= d_k
        if abs(d_cx - 0.5) < 2e-4 and abs(d_cy - 0.5) < 2e-4 and abs(d_k - 1.0) < 2e-3:
            break
    cam_data.clip_start = 0.1
    cam_data.clip_end = d_dist + 20.0
    return cam


def setup_render(scene):
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for s_backend in ("METAL", "OPTIX", "CUDA", "HIP"):
        try:
            prefs.compute_device_type = s_backend
        except TypeError:
            continue
        prefs.get_devices()
        if any(d.type != "CPU" for d in prefs.devices):
            for d in prefs.devices:
                d.use = True
            scene.cycles.device = "GPU"
            break
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    # The real point of the cooker is a caustic: 1284 mirrors concentrating
    # the sun on the dog. Traced as one, it is noise the denoiser smears,
    # and turning it off changes the picture by less than the noise did.
    scene.cycles.caustics_reflective = False
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.resolution_percentage = PCT
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.compression = 100
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"


def main():
    clear_scene()
    add_plate()
    add_mirrors()
    add_dog()
    add_lights()
    scene = bpy.context.scene
    setup_render(scene)
    add_camera()

    if bpy.app.background:
        scene.render.filepath = OUT
        print(f"[render] cycles {RES_X}x{RES_Y} {SAMPLES} spp device={scene.cycles.device}")
        bpy.ops.render.render(write_still=True)
        print("WROTE", OUT)
        return

    def _setup_viewport():
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == "VIEW_3D":
                    for space in area.spaces:
                        if space.type == "VIEW_3D":
                            space.shading.type = "RENDERED"
                            space.region_3d.view_perspective = "CAMERA"

    bpy.app.timers.register(_setup_viewport, first_interval=0.4)


main()
