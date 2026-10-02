"""Headless render of the primitives-showcase board (reuses the GUI layout).

    RENDER_SCALE=1 /Applications/Blender.app/Contents/MacOS/Blender \
        --background --python render.py                      # 1620 x 2160 preview
    SHOWCASE_THEME=light RENDER_SCALE=8 RENDER_TILES=2 \
        /Applications/Blender.app/Contents/MacOS/Blender \
        --background --python render.py                      # ARCH E master

Writes to ``renders/`` (gitignored). Downscale the light master into
``docs/img/primitives_showcase.png`` for the docs (see this package's
README).
"""

from __future__ import annotations

import sys
from pathlib import Path

SHOWCASE_DIR = Path(__file__).resolve().parent
# Blender runs this file as a bare script (`--python render.py`), with no
# package context, so `sdm_showcase` is not importable until its parent
# (`src/`) is on sys.path. Editable-installing the package into Blender's own
# Python would remove the need for this, but would also make the standalone
# `--python render.py` invocation in this file's docstring depend on a prior
# install step -- this keeps it a single command.
_SRC_DIR = str(SHOWCASE_DIR.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from sdm_showcase.layout import build_layout  # noqa: E402 (needs the sys.path shim above)


def main() -> None:
    """Build the board, then render it headlessly to a PNG via EEVEE.

    ``bpy`` only exists inside Blender's embedded Python, so it is imported
    here rather than at module scope: see :func:`sdm_showcase.layout.build_layout`'s
    docstring for why this matters beyond this one script.
    """
    try:
        import bpy
    except ImportError as exc:
        raise RuntimeError(
            "render.py must run inside Blender's Python (bpy is a "
            "Blender-embedded module, not pip-installable). Run it as "
            "`blender --background --python render.py`."
        ) from exc

    import os

    build_layout()

    sc = bpy.context.scene
    sc.render.engine = "BLENDER_EEVEE"
    scale = int(os.environ.get("RENDER_SCALE", "1"))
    # Portrait page at layout.py's POSTER_ASPECT. 2160 px tall at 1x makes 4x
    # a 24 in edge at 360 dpi, so RENDER_SCALE=4 is already a print master.
    from sdm_showcase.layout import POSTER_ASPECT

    res_y = 2160 * scale
    res_x = int(round(2160 * POSTER_ASPECT)) * scale
    # A single EEVEE frame cannot exceed the GPU's texture edge (16384 px on
    # Apple silicon); beyond it Blender writes a fully transparent image with
    # no error. RENDER_TILES=N renders an N x N grid of sub-frames by shifting
    # the orthographic camera, each under the limit, and stitches them with
    # ImageMagick if it is on PATH (else the tiles are left for the caller).
    # RENDER_SCALE=8 RENDER_TILES=2 is a 12960 x 17280 ARCH E master at 360 dpi.
    tiles = int(os.environ.get("RENDER_TILES", "1"))
    if res_x % tiles or res_y % tiles:
        raise ValueError(f"RENDER_TILES={tiles} must divide {res_x} x {res_y}")
    sc.render.resolution_x = res_x // tiles
    sc.render.resolution_y = res_y // tiles
    if max(sc.render.resolution_x, sc.render.resolution_y) > 16384:
        raise ValueError(
            f"a {sc.render.resolution_x} x {sc.render.resolution_y} tile exceeds the 16384 px "
            "GPU limit and would render blank; raise RENDER_TILES"
        )
    try:
        sc.eevee.taa_render_samples = 192  # clean edges at high res
    except AttributeError as exc:
        # Blender version without this EEVEE setting: render still proceeds,
        # just without the extra sample count.
        print("taa set failed:", exc)

    out_dir = SHOWCASE_DIR / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_light" if os.environ.get("SHOWCASE_THEME", "dark").lower() == "light" else ""
    out_path = out_dir / f"primitives_showcase{suffix}_{scale}x.png"
    sc.render.image_settings.file_format = "PNG"
    if tiles == 1:
        sc.render.filepath = str(out_path)
        bpy.ops.render.render(write_still=True)
        print("WROTE", sc.render.filepath, f"({res_x}x{res_y})")
        return

    # Blender measures ortho_scale and shift against the frame's larger side
    # (the height, for a portrait page). A tile is 1/N of the page in both
    # directions, so its shift, in units of its own ortho_scale, is the tile's
    # offset from the page centre: whole steps vertically, aspect-scaled steps
    # horizontally. Rows run top to bottom so the stitch order reads naturally.
    cam = sc.camera.data
    cam.ortho_scale = cam.ortho_scale / tiles
    tile_paths: list[list[str]] = []
    for row in range(tiles):
        tile_paths.append([])
        for col in range(tiles):
            cam.shift_x = (col - (tiles - 1) / 2.0) * POSTER_ASPECT
            cam.shift_y = (tiles - 1) / 2.0 - row
            tile = out_dir / f"{out_path.stem}_r{row}c{col}.png"
            sc.render.filepath = str(tile)
            bpy.ops.render.render(write_still=True)
            tile_paths[-1].append(str(tile))
            print("TILE", tile)
    import shutil
    import subprocess

    magick = shutil.which("magick")
    if magick is None:
        print(f"LEFT {tiles * tiles} tiles in {out_dir}; join rows with +append, then -append")
        return
    # Each row is joined left to right, then the rows top to bottom.
    cmd = [magick]
    for row_paths in tile_paths:
        cmd += ["(", *row_paths, "+append", ")"]
    cmd += ["-append", str(out_path)]
    subprocess.run(cmd, check=True)
    for row_paths in tile_paths:
        for tile_path in row_paths:
            Path(tile_path).unlink()
    print("WROTE", out_path, f"({res_x}x{res_y}, stitched from {tiles}x{tiles} tiles)")


if __name__ == "__main__":
    main()
