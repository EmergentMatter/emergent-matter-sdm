"""Build a disco-lens part as a ``.sdm``.

    disco-lens-build --out quadrant.sdm --set mirror_size=10.2

Both ``--out`` and ``--set`` are required by the viewer's ``/reemit``
contract. ``--set`` alone fails inside a subprocess, and the error
surfaces nowhere useful.

By default this REUSES the solved aim heights in
``layout/mirror_table.csv``, which is the design of record.
``--optimize`` re-solves them instead. Both are fast (the solve is
about 1.4 s) and both are deterministic, so the difference is not cost:
it is that the table pins the shipped design, while ``--optimize`` is
what you need when ``--set`` changes the layout and the table no longer
describes it.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

from disco_lens.design_io import MIRROR_TABLE, sites_from_table
from disco_lens.harness import save_sdm
from disco_lens.parameters import DiscoLensParameters

__all__ = ["main"]

#: Part name to the builder that makes it. The ring carries no sites,
#: so it is the one part that never needs the mirror table.
PARTS = {
    "quadrant": ("build_quadrant_part", True),
    "ring": ("build_ring_part", False),
}


def _coerce(s_field: str, s_raw: str) -> bool | int | float | str:
    """Cast by the prefix, which IS the type in this codebase.

    An ``n_`` field parses through float FIRST. The ``.sdm`` stores every
    param value as a JSON number, so an integer round-trips as ``8.0``; a
    viewer reads that and hands it straight back on re-emit, and
    ``int("8.0")`` raises. The failure reads "expects a whole number, got
    '8.0'", which looks like a whole number and sends you looking in the
    wrong place. A genuinely non-integral value is still refused.
    """
    if s_field.startswith("n_"):
        d_value = float(s_raw)
        if d_value != int(d_value):
            raise ValueError(f"{s_raw} is not a whole number")
        return int(d_value)
    if s_field.startswith("b_"):
        return s_raw.lower() in ("1", "true", "yes")
    if s_field.startswith("s_"):
        return s_raw
    return float(s_raw)


def _emitted_aliases(known: set[str]) -> dict[str, str]:
    """The name the document ships to the field it came from.

    A viewer reads the EMITTED name and passes that back on re-emit, so
    a CLI that only knows field names refuses every rebuild a viewer
    sends. Having --out and --set is necessary but not sufficient: the
    names have to agree in both directions.
    """
    from disco_lens.sdf_assembly import PARAM_FIELDS

    aliases: dict[str, str] = {}
    # Prefix stripping is only a convenience for fields the document
    # does not carry. For the ones it DOES carry, the emitted name is
    # whatever PARAM_FIELDS says, which is not always the field minus
    # its prefix: d_plate_size ships as assembled_plate_size, because
    # the document describes the assembled cooker and the param
    # describes one plate of it.
    for s_field in known:
        s_alias = s_field.split("_", 1)[1]
        if s_alias in aliases:
            raise AssertionError(
                f"two fields emit the same name {s_alias!r}: "
                f"{aliases[s_alias]!r} and {s_field!r}. The .sdm could "
                "not tell them apart."
            )
        aliases[s_alias] = s_field
    for s_field, s_emits, _ in PARAM_FIELDS:
        aliases[s_emits] = s_field
    return aliases


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="disco-lens-build", description="Build a disco-lens part as a .sdm"
    )
    ap.add_argument("--out", required=True, type=Path, help="output .sdm path")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override a parameter; repeatable",
    )
    ap.add_argument(
        "--part",
        default="quadrant",
        choices=sorted(PARTS),
        help="which part to build (default: quadrant)",
    )
    ap.add_argument(
        "--validate-schema",
        action="store_true",
        help="run the full jsonschema pass on the way out "
        "(301 s on the quadrant, byte-identical result; "
        "the default is a round-trip check instead)",
    )
    ap.add_argument(
        "--optimize",
        action="store_true",
        help="re-solve the aim heights (~1.5 s) instead of "
        "reusing the solved mirror table; required when "
        "--set changes the mirror layout",
    )
    args = ap.parse_args(argv)

    known = {f.name for f in dataclasses.fields(DiscoLensParameters)}
    aliases = _emitted_aliases(known)

    overrides: dict[str, Any] = {}
    for s_item in args.set:
        if "=" not in s_item:
            ap.error(f"--set expects NAME=VALUE, got {s_item!r}")
        s_key, s_raw = s_item.split("=", 1)
        s_key = aliases.get(s_key, s_key)
        if s_key not in known:
            # Naming the near miss turns a typo into a one-line fix
            # instead of a silently ignored override.
            near = sorted(k for k in known if k.split("_", 1)[-1] == s_key.split("_", 1)[-1])
            ap.error(
                f"unknown parameter {s_key!r}" + (f": did you mean {near[0]!r}?" if near else "")
            )
        try:
            overrides[s_key] = _coerce(s_key, s_raw)
        except ValueError:
            s_kind = "whole number" if s_key.startswith("n_") else "number"
            ap.error(f"{s_key!r} expects a {s_kind}, got {s_raw!r}")

    p = dataclasses.replace(DiscoLensParameters(), **overrides)
    problems = p.validate()
    if problems:
        # Non-zero exit AND a named cause on stderr. A viewer's re-emit
        # reads proc.stderr to explain a failure and DISCARDS stdout, so
        # printing the reason to stdout surfaces in the browser as
        # "authoring failed:" with nothing after it.
        print(
            "failed: invalid parameters:\n  " + "\n  ".join(problems), file=sys.stderr, flush=True
        )
        return 2

    from disco_lens import sdf_assembly

    s_builder, b_needs_sites = PARTS[args.part]
    build = getattr(sdf_assembly, s_builder)
    try:
        if not b_needs_sites:
            part = build(p)
        elif args.optimize:
            from disco_lens.optics import optimize_targets

            part = build(p, optimize_targets(p))
        else:
            part = build(p, sites_from_table(p, MIRROR_TABLE))
    except (FileNotFoundError, KeyError) as e:
        print(
            f"failed: {e}\n  pass --optimize to solve new aims for this design",
            file=sys.stderr,
            flush=True,
        )
        return 2

    recorded = [sys.argv[0]] + (list(argv) if argv is not None else sys.argv[1:])
    path = save_sdm(part, out=args.out, argv=recorded, b_validate_schema=args.validate_schema)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
