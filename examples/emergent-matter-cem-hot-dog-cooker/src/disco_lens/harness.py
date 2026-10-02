"""Save a ``Part`` to a ``.sdm``, stamping how it was produced.

Thin on purpose. What it adds over ``software_defined_matter.save`` is
GENERATOR METADATA: the argv and cwd that made the file. A viewer's
``/reemit`` reads exactly that to rebuild the part when a param changes.
Without it the document can be rendered and never re-derived.

The ``/reemit`` contract is ``--out`` AND ``--set``. A CEM whose CLI
takes only ``--set`` fails inside a subprocess, where the error is easy
to lose entirely.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from software_defined_matter import Part

__all__ = ["NON_REPLAYABLE_FLAGS", "save_sdm", "save_verified"]

#: Flags stripped from the recorded generator command.
#:
#: THE RECORDED COMMAND MUST BE CHEAP AND IDEMPOTENT. A viewer replays
#: it synchronously on every parameter release, so whatever it does is
#: latency a person feels, and a command that answers differently each
#: time reads to the viewer as a change it must chase.
#:
#: ``--validate-schema`` is the one flag here that fails the first test
#: and not the second. Measured on the quadrant it adds 301 s and
#: produces a byte-identical file, so replaying it would blow the
#: viewer's budget by two orders of magnitude to change nothing.
#:
#: ``--optimize`` is deliberately NOT here, though an earlier draft of
#: this file had it on the theory that re-solving is expensive. It is
#: not: the aim solve is 1.4 s for 1284 sites and deterministic. It is
#: also load-bearing for replay. A document built with ``--set`` on a
#: param that changes the layout can ONLY be rebuilt by re-solving,
#: because the recorded mirror table describes a different field, so
#: stripping the flag would record a command that exits 2.
NON_REPLAYABLE_FLAGS: frozenset[str] = frozenset({"--validate-schema"})


def _replayable_argv(argv: list[str]) -> list[str]:
    """The command minus anything that would not reproduce this file."""
    return [a for a in argv if a not in NON_REPLAYABLE_FLAGS]


def save_verified(part: Part, out: Path, *, b_validate_schema: bool = False) -> Path:
    """Write ``part`` and prove the file reads back.

    The schema pass is OFF by default, and the round trip is what
    replaces it. Measured on the quadrant, a 2.1 MB document: saving
    with ``b_validate_schema=True`` takes 301 s, saving without it takes
    0.04 s, and the two produce byte-identical files. Loading the file
    back costs nothing and catches what actually goes wrong, which is
    the writer emitting something that will not parse.

    That is also the template's own advice. ``schema_ok: True`` means
    the JSON is shaped right and says nothing about whether the geometry
    is what you meant, so five minutes is a lot to pay for it on every
    build. Pass ``b_validate_schema=True`` when the point IS the wire
    contract, such as before publishing an artifact for another tool.
    """
    from software_defined_matter import load, save

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save(part, out, b_validate_schema=b_validate_schema)
    load(out, b_validate_schema=False)
    return out


def save_sdm(
    part: Part, *, out: Path, argv: list[str] | None = None, b_validate_schema: bool = False
) -> Path:
    """Serialise ``part`` to ``out``, recording how to rebuild it."""
    part.metadata = dict(part.metadata or {})
    part.metadata["generator"] = {
        # Defaults to sys.argv, which is right for a real CLI call and
        # wrong whenever main() runs in-process: under pytest sys.argv
        # is pytest's own command line, so the document would record a
        # generator that rebuilds nothing. The CLI passes its own.
        "argv": _replayable_argv(list(argv) if argv is not None else list(sys.argv)),
        "cwd": str(Path.cwd()),
        # Which flags this CLI understands. A tool cannot tell that from
        # argv alone; undeclared, its only option is to write edited
        # values into the .sdm and hope the generator reads its own
        # output. Both together or neither: --set without --out would
        # have the generator write over its source mid-rebuild.
        "accepts": ["--out", "--set"],
    }
    return save_verified(part, out, b_validate_schema=b_validate_schema)
