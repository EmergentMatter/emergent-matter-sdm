"""Export the quadrant plate + clamp ring to STLs for slicing.

    uv run python scripts/export_stl.py [voxel_size_mm]

Print the quadrant FOUR times. Repair is deterministic cleanup only:
it drops zero-volume debris and patches pinholes, and fails loudly if
that would reshape the part.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from software_defined_matter.export import export_part

import disco_lens.sdm_ext  # noqa: F401  (single-wedge fold fast path)

ROOT = Path(__file__).resolve().parent.parent


def archive_older(path: Path) -> int:
    """Move every OLDER stl of this part family into stl/archive/.

    Rob's rule (2026-08-19): only the current version of each part
    stays in stl/ - stale timestamps in a mesh review tool's dropdown
    are how you end up marking up last week's geometry. Nothing is
    deleted.
    """
    d_dir = path.parent
    d_arch = d_dir / "archive"
    s_family = re.sub(r"_\d{4}-\d{2}-\d{2}T\d{4}\.stl$", "", path.name)
    n_moved = 0
    for f in d_dir.glob(f"{s_family}_*.stl"):
        if f.name == path.name:
            continue
        d_arch.mkdir(exist_ok=True)
        shutil.move(str(f), str(d_arch / f.name))
        n_moved += 1
    if n_moved:
        print(f"{path.name}: archived {n_moved} older {'file' if n_moved == 1 else 'files'}")
    return n_moved


def repair(path: Path) -> None:
    """Deterministic cleanup, no black-box remeshing: keep the largest
    component (drops zero-volume debris like feather specks), patch
    pinholes, verify against the pre-repair main body."""
    mesh = trimesh.load(path)
    if mesh.is_watertight:
        print(f"{path.name}: watertight")
        return
    comps = mesh.split(only_watertight=False)
    main = max(comps, key=lambda c: len(c.faces))
    d_vol0, a_bb0 = float(main.volume), main.bounds.copy()
    main.merge_vertices()
    main.update_faces(main.nondegenerate_faces())
    main.update_faces(main.unique_faces())
    main.remove_unreferenced_vertices()
    main.fill_holes()
    d_dvol = abs(float(main.volume) - d_vol0) / max(d_vol0, 1e-9)
    d_dbb = float(np.abs(main.bounds - a_bb0).max())
    if d_dvol > 0.005 or d_dbb > 0.05:
        raise SystemExit(
            f"{path.name}: repair changed the part (vol {d_dvol * 100:.2f}%, bbox {d_dbb:.2f})"
        )
    main.export(path)
    s_state = "watertight" if main.is_watertight else "STILL LEAKY"
    print(f"{path.name}: {len(comps) - 1} debris specks dropped -> {s_state}")


def main() -> None:
    d_voxel = float(sys.argv[1]) if len(sys.argv) > 1 else 0.35
    for s_name in ("disco_lens_quadrant.sdm", "disco_lens_ring.sdm"):
        d_v = min(d_voxel, 0.2) if "ring" in s_name else d_voxel
        paths = export_part(ROOT / s_name, ROOT / "stl", voxel_size=d_v)
        for path in paths:
            print(f"wrote {path}")
            repair(Path(path))
            archive_older(Path(path))


if __name__ == "__main__":
    main()
