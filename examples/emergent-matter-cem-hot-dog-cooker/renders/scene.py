"""Write ``scene.json``: everything ``render.py`` places besides the plate mesh.

Nothing here is needed to build the part. Blender's Python cannot import
this package (no JAX, no sdm-core), so this half runs under ``uv`` and
hands Blender plain numbers:

    uv run python renders/scene.py
    blender --background --python renders/render.py

Mirrors are posed from the design of record (``layout/mirror_table.csv``),
re-solved exactly as the ray trace does, so the glass in the picture sits
where the build puts it. Each tile is its pad's tilt, the conjugation
``Rz(az)·Ry(-a)·Rz(-az)`` from ``sdf_assembly``: it leans toward the dog
while its edges stay along the plate axes.

The plate itself comes from the newest quadrant and ring STL in ``stl/``,
so run ``scripts/export_stl.py`` first.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from disco_lens.design_io import MIRROR_TABLE, sites_from_table
from disco_lens.parameters import DiscoLensParameters

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "scene.json"


def newest_stl(s_part: str) -> Path:
    """The current export of one part; ``export_stl.py`` archives the rest."""
    l_found = sorted((ROOT / "stl").glob(f"disco_lens_{s_part}_*.stl"))
    if not l_found:
        raise FileNotFoundError(f"no {s_part} STL in stl/: run scripts/export_stl.py first")
    return l_found[-1]


def tile_rotation(d_az: float, d_alpha: float) -> list[list[float]]:
    """Row-major 3x3 for ``Rz(az)·Ry(-alpha)·Rz(-az)``, geometry sense."""
    d_c, d_s = math.cos(d_az), math.sin(d_az)
    d_ca, d_sa = math.cos(d_alpha), math.sin(d_alpha)

    def rz(d_cz: float, d_sz: float) -> list[list[float]]:
        return [[d_cz, -d_sz, 0.0], [d_sz, d_cz, 0.0], [0.0, 0.0, 1.0]]

    ry = [[d_ca, 0.0, -d_sa], [0.0, 1.0, 0.0], [d_sa, 0.0, d_ca]]

    def mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

    return mul(mul(rz(d_c, d_s), ry), rz(d_c, -d_s))


def main() -> None:
    p = DiscoLensParameters()
    l_mirrors = []
    for site in sites_from_table(p, MIRROR_TABLE):
        l_r = tile_rotation(site.d_azimuth, site.d_tilt)
        l_n = [l_r[0][2], l_r[1][2], l_r[2][2]]
        # d_mirror_z is the OPTICAL surface; the tile's centre is half a
        # thickness back along its normal.
        d_back = 0.5 * p.d_mirror_thickness
        l_mirrors.append(
            {
                "center": [
                    site.d_x - l_n[0] * d_back,
                    site.d_y - l_n[1] * d_back,
                    site.d_mirror_z - l_n[2] * d_back,
                ],
                "rotation": l_r,
            }
        )

    d_ring_z = p.d_hub_height - p.d_ring_height  # flush with the post top
    scene = {
        "quadrant_stl": str(newest_stl("quadrant")),
        "ring_stl": str(newest_stl("ring")),
        "ring_z": d_ring_z,
        "mirror_size": p.d_mirror_size,
        "mirror_thickness": p.d_mirror_thickness,
        "mirrors": l_mirrors,
        "dog": {
            "radius": p.d_dog_radius,
            "bottom": p.d_dog_bottom_height,
            "top": p.d_dog_top_height,
        },
        "skewer": {
            "radius": 2.0,
            "bottom": p.d_hub_height - 30.0,
            "top": p.d_dog_top_height + 40.0,
        },
    }
    OUT.write_text(json.dumps(scene, indent=1))
    print(f"wrote {OUT} ({len(l_mirrors)} mirrors)")


if __name__ == "__main__":
    main()
