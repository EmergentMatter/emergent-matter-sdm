"""Read a solved design back out of its mirror table.

Everything downstream of the aim solve (the ray trace, the CLI's
default path) reads the solved aims back from
``layout/mirror_table_v*.csv`` rather than re-solving.

Not for speed: the solve is about 1.4 s. The table is the design OF
RECORD. It is what the shipped ``.sdm`` was built from, what the
assembly instructions send you to when gluing 1284 mirrors, and the
only copy of the optimizer's choices that survives outside a process.
Reading it back means the exporters and the ray trace describe the part
on the bench, not a fresh solve that happens to agree.

That is sound because the layout and the per-site tilt solve are both
deterministic: given the same parameters, ``layout_sites`` produces the
same sites in the same order, and re-solving each one against its
recorded aim height reproduces the tilt exactly. What the table carries
is the one thing that cannot be recomputed cheaply, which is which aim
height the optimizer chose for each group.
"""

from __future__ import annotations

import csv
from pathlib import Path

from disco_lens.optics import Site, layout_sites, solve_site
from disco_lens.parameters import DiscoLensParameters

__all__ = ["MIRROR_TABLE", "sites_from_table"]

ROOT = Path(__file__).resolve().parents[2]

#: The mirror table the build script writes; the design of record.
MIRROR_TABLE = ROOT / "layout" / "mirror_table.csv"


def sites_from_table(p: DiscoLensParameters, path: Path | str) -> list[Site]:
    """Rebuild the solved site list for ``p`` from a mirror table.

    Sites are laid out fresh from ``p`` and matched to the table by
    position, so a table written for a different plate does not quietly
    half-apply: the lookup raises instead.

    Raises:
        FileNotFoundError: If the table has not been built yet.
        KeyError: If ``p`` lays out a site the table does not cover,
            which means the table belongs to a different design and the
            aims have to be re-optimized.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist: run the build script for this version "
            "once before anything can reuse its aims"
        )

    aims: dict[tuple[float, float], float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            aims[(round(float(row["x_mm"]), 2), round(float(row["y_mm"]), 2))] = float(
                row["aim_height_mm"]
            )

    sites = layout_sites(p)
    keys = {(round(s.d_x, 2), round(s.d_y, 2)) for s in sites}

    # Covering the layout is NOT enough to make the aims reusable, and
    # assuming it was is how this quietly built a wrong part. A smaller
    # plate on the same pitch lays out a strict SUBSET of a bigger
    # plate's sites, so every lookup succeeds and every aim comes back:
    # aims that were optimized to flatten a different, wider field.
    # Nothing errors and the document is plausible. The set has to match
    # exactly, because that is what says the table is this design's.
    if keys != set(aims):
        n_extra, n_missing = len(set(aims) - keys), len(keys - set(aims))
        raise KeyError(
            f"{path.name} describes a different design: it carries "
            f"{len(aims)} sites and this one lays out {len(sites)} "
            f"({n_missing} not in the table, {n_extra} in the table and "
            "not in the layout). Its aims were optimized for that field, "
            "not this one, so reusing them would build a plausible part "
            "with the wrong aim heights"
        )

    for s in sites:
        solve_site(p, s, aims[(round(s.d_x, 2), round(s.d_y, 2))])
    return sites
