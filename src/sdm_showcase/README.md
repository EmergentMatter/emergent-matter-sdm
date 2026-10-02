# Primitives Showcase

A single rendered board of **every `sdm-core` primitive**, each shown with
three parameter perturbations, color-coded by family, with a combined Lofts
& Sweeps row at the bottom. A concrete demonstrator for `sdm-core`'s
geometry: it needs only `sdm-core` and Blender, and imports nothing from
`sdm-view`.

Family names, colors, and per-primitive parameter sweeps are defined once,
in `build.py`'s `build_catalog()` and `build_constructions()`: read those
rather than this file for the current catalog contents, so this doc can't
go stale against a renamed or added primitive.

## Themes

Two looks from the same geometry, selected with the `SHOWCASE_THEME` env
var: `dark` (default) or `light`. See `layout.py`'s `THEME`, `LIGHT`, and
`LIGHT_PALETTE` for exactly what each one changes.

Both themes share the same flat orthographic camera: axis-aligned, looking
straight down +Y with up = +Z and zero roll, so there is no perspective
convergence and every part sits on the same picture plane. The themes
differ only in background, text, and color treatment.

To keep the camera (and therefore the page/text layout) dead-on while
still letting the solid families read as 3-D, the parts themselves are
tilted a few degrees about two axes (see `build.py`'s `tilt()` and
`TILT_CATS`). Only the solid families tilt; the flat 2-D profiles and the
lofts/sweeps band stay axis-aligned.

## Pipeline

`sdm-core` does the geometry, Blender does the pixels.

1. **`build.py`** builds the SDF tree for each catalog entry, meshes it,
   and writes an STL plus `manifest.json` into `mega/` (gitignored).
2. **`layout.py`** imports the STLs per the manifest, color-codes them,
   adds labels and vertical family legends, and sets up the camera, lights,
   and theme. Used both for interactive framing in Blender's GUI and by the
   headless renderer below.
3. **`render.py`** imports `layout.py`'s `build_layout()`, then renders a
   portrait PNG into `renders/` (gitignored) via EEVEE.

Each script's module docstring carries its own exact invocation, since
that is what actually needs to stay in sync with its arguments; this
section is the map between them, not a copy of the commands.

`render.py` frames a portrait page at `layout.py`'s `POSTER_ASPECT` (3:4 by
default: ARCH E and the 18 x 24 in family; set `POSTER_ASPECT=0.6667` for
ARCH D / 24 x 36 in) with equal margins computed from the content, and sizes
the pixels from the same aspect: 1620 x 2160 at `RENDER_SCALE=1`, so
`RENDER_SCALE=4` is a 24 in edge at 360 dpi and `RENDER_SCALE=6` a 36 in edge
at 360 dpi. `SHOWCASE_THEME=light` renders the white poster.

One EEVEE frame cannot exceed the GPU's texture edge (16384 px on Apple
silicon); past it Blender writes a fully transparent PNG and reports
success. `RENDER_TILES=N` renders an N x N grid of camera-shifted sub-frames
instead and stitches them with ImageMagick (`magick` on PATH; without it the
tiles are left in `renders/`). The ARCH E master is

```bash
SHOWCASE_THEME=dark RENDER_SCALE=8 RENDER_TILES=2 \
    /Applications/Blender.app/Contents/MacOS/Blender --background --python render.py
```

which is 12960 x 17280 px: 360 dpi at 36 x 48 in, 270 dpi at 48 x 64 in.

The last step, downscaling the print master into the docs asset this
package's own consumer embeds, is not part of any script (it is a one-off
publish step, not part of regenerating the board itself). The docs use the
white poster; at 1050 x 1400 it is about 380 KB in truecolour, under the
org's 500 KB image budget, so it is not palette-quantised (the dark board
would need `+dither -colors 256` to fit, and its soft shadows would not
survive that anyway):

```bash
magick renders/primitives_showcase_light_8x.png -filter Lanczos -resize 1050x1400 \
    -strip -define png:compression-level=9 ../../docs/img/primitives_showcase.png
```

`mega/` and `renders/` are gitignored (see `.gitignore` in this
directory); only the scripts are versioned, and everything else
regenerates from the commands in their docstrings.

## Capability requirements

This showcase depends on geometry capabilities in `sdm-core` beyond what
every checkout of it provides. Restated here as requirements, not as
citations into `sdm-core`'s own PR history (which this repo's readers may
not have access to): if `build.py` fails or a primitive is missing from
the board, check these against the `sdm-core` checkout in use.

- **Loft and sweep primitives.** The Lofts & Sweeps row needs
  `sdf_loft(..., interp="shape")` (curve-to-curve morphing) and
  `sdf_sweep(..., path_kind=...)` (profile-along-path, polyline or
  bspline).
- **A correctly signed `pyramid` SDF.** `build.py` uses `sdm-core`'s real
  `pyramid` primitive directly (no workaround), plus an orientation
  transform for camera-facing framing. A `pyramid` whose interior distance
  field is single-signed never crosses zero, so it meshes as nothing (or
  renders as a flat square) instead of a solid. Current `sdm-core`
  checkouts mesh it as a solid; the shipped board predates that.
- **`export_part` and `grid_sampling.BBoxResolutionError`.** Every cell is
  meshed through sdm-core's public exporter, wrapped as a one-material
  `Part`; nothing here imports a private submodule any more. The exporter
  bounds each tree itself, and where its analytic bounder declines (an
  extrusion whose child is a modifier, which is every 2-D profile ribbon
  here) `build.py` retries with an explicit `metadata["bbox"]`, the escape
  hatch its error message names.
- **TPMS primitives that take `period` and `min_thickness`.** The five
  lattice entries pass a spatial period in mm and a physical minimum wall
  thickness in mm, the signature `sdm-core` adopted when it made its
  primitives return conservative distances. An older checkout that still
  takes `scale` and `thickness` fails every TPMS cell with a `TypeError`.

## Known limitations

- **TPMS and thread cells mesh on their own finer grid.** Their thinnest
  features sit near 0.3 mm, below the 0.4 mm voxel the rest of the board
  uses, and marching cubes shreds a wall thinner than about two voxels
  into disconnected flakes. `build.py` meshes those cells at `TPMS_VOXEL`;
  sdm-core bounds each one tightly from its own tree, so they cost what
  their 18 mm cubes cost, but they still write STLs of 60 to 90 MB each.
  `mega/` is gitignored for a reason.
- **Light-theme drop shadows** come from a white backdrop plane just behind
  the parts and a single soft sun tilted off the view axis, with the world
  fill dimmed so shadowed paper reads as grey and the sun sized so open
  paper is exactly white; the dark theme's area lights are off in that mode
  because any second lamp lifts the shadow back to white. EEVEE's per-light
  shadow texel floor (`shadow_maximum_resolution`, 1 mm by default) has to
  be lowered far below the part scale or every shadow renders as a block the
  size of its caster. See `layout.py`'s `LIGHT_*` constants.
- **Light-theme rendering** uses Blender's *Standard* view transform
  rather than AgX (AgX greys a white world and desaturates the catalog
  colors), hand-desaturates the saturated catalog hues for the white
  poster, exposes parts near their albedo so they do not clip into neon
  against the white world, and zeroes specular on label materials so
  black type reads as true black. See `layout.py`'s `LIGHT_PALETTE`,
  `material_for()`, and `flatten_text()`.
