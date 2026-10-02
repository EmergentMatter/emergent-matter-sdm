#!/usr/bin/env python3
"""Check this CEM against the contract a TOOL depends on. No browser needed.

    uv run python scripts/check_contract.py

WHY THIS EXISTS. Building one real CEM turned up fourteen defects. Ten were
found by *running* something rather than by testing, and four specifically by
pointing a viewer at the part and dragging a slider. The repo was green
throughout: every one of those four produced a document that built, validated,
and was wrong in a way only a tool could see.

Five of them are checkable right here, in seconds, with no browser:

  * every param carries a unit and a usable range
  * the CLI accepts the name the document EMITS, not just the field name
  * an integer param survives a float round-trip (`8.0`, not `8`)
  * failures go to stderr with a non-zero exit
  * the recorded generator is idempotent and reproduces the document
  * the recorded generator is CHEAP enough to sit under a slider, and writes
    nothing but its --out

The sixth, whether a viewer renders it, still needs a viewer.

The last two are about cost, not correctness. A generator can re-run and
reproduce its document exactly and still be unusable: re-emit replays the
recorded command while the viewer blocks on it, so an authoring command that
meshes or verifies pays that on every slider release. From the browser an
expensive generator is indistinguishable from "the parameter was ignored",
which is why correctness alone is not enough to check.

This drives the CLI as a SUBPROCESS, the way a tool does, rather than calling
`main()` in-process. That difference is not cosmetic: an in-process call shares
`sys.argv`, swallows nothing, and hides exactly the stream and exit-code bugs
this is looking for.

Stdlib only, so it runs before anything is installed except the package itself.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OK, BAD = "  ok   ", " FAIL  "

#: Wall-clock ceiling for one replay of the recorded generator, seconds.
#:
#: The viewer runs it SYNCHRONOUSLY on every topology-param release
#: (`subprocess.run`, blocking), so this is the latency a person feels when
#: they let go of a slider. 5 s is deliberately generous: a CEM that only
#: authors a tree lands in 1-3 s, and anything an order of magnitude over that
#: is doing work that belongs behind a flag: meshing, verifying, optimising,
#: calling a solver. `docs/traps.md` covers what to do about it.
SLIDER_BUDGET_S = 5.0

_fails: list[str] = []


def check(b_ok: bool, s_label: str, s_detail: str = "") -> bool:
    print((OK if b_ok else BAD) + s_label + (f"  {s_detail}" if s_detail else ""))
    if not b_ok:
        _fails.append(s_label)
    return b_ok


def manifest() -> dict:
    with (ROOT / "cem.toml").open("rb") as fh:
        return tomllib.load(fh)


def run(args: list[str]) -> subprocess.CompletedProcess:
    """Invoke the CEM's entry point the way a tool would."""
    return subprocess.run(["uv", "run", SCRIPT, *args], cwd=ROOT, capture_output=True, text=True)


def entry_script() -> str:
    """The console script this CEM installs, from pyproject.

    Read rather than hard-coded so this file copies verbatim into any CEM,
    which is the whole point of it.
    """
    with (ROOT / "pyproject.toml").open("rb") as fh:
        scripts = tomllib.load(fh).get("project", {}).get("scripts", {})
    if not scripts:
        raise SystemExit("pyproject declares no [project.scripts]; a tool has nothing to call")
    return next(iter(scripts))


SCRIPT = entry_script()


def main(argv: list[str] | None = None) -> int:
    if not shutil.which("uv"):
        print("error: uv is not installed; run scripts/doctor.py", file=sys.stderr)
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="cem-contract-"))
    try:
        return _checks(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _checks(tmp: Path) -> int:
    m = manifest()
    params = m["cem"]["params"]

    print("\nCEM tool contract\n")

    # --- it builds at all ---------------------------------------------------
    out = tmp / "base.sdm"
    p = run(["--out", str(out)])
    if not check(
        p.returncode == 0 and out.exists(),
        "builds with --out",
        "" if p.returncode == 0 else p.stderr.strip()[:120],
    ):
        return _summary()
    doc = json.loads(out.read_text())

    # --- every param is controllable ----------------------------------------
    missing_unit = [k for k, v in doc["params"].items() if not v.get("unit")]
    check(
        not missing_unit,
        "every param carries a unit",
        "" if not missing_unit else f"missing on {missing_unit[:4]}",
    )

    unbounded = [k for k, v in doc["params"].items() if v.get("bounds") is None]
    check(
        not unbounded,
        "every param carries bounds",
        ""
        if not unbounded
        else f"{len(unbounded)} unbounded, e.g. {unbounded[:3]} -- these render as "
        "controls nothing can move",
    )

    wide = [
        k
        for k, v in doc["params"].items()
        if v.get("bounds") and v["bounds"][0] > 0 and v["bounds"][1] / v["bounds"][0] > 200
    ]
    check(
        not wide,
        "bounds are usable, not merely valid",
        "" if not wide else f"{wide[:3]} span >200x; a slider over that is unusable",
    )

    # --- the emitted-name contract ------------------------------------------
    emitted = {v.get("emits"): k for k, v in params.items() if v.get("emits")}
    s_name = next(iter(emitted), None)
    if s_name:
        field = emitted[s_name]
        d_val = params[field]["default"]
        r = run(["--out", str(tmp / "a.sdm"), "--set", f"{s_name}={d_val}"])
        check(
            r.returncode == 0,
            f"--set accepts the EMITTED name ({s_name})",
            "" if r.returncode == 0 else r.stderr.strip().splitlines()[-1][:110],
        )
        r = run(["--out", str(tmp / "b.sdm"), "--set", f"{field}={d_val}"])
        check(
            r.returncode == 0,
            f"--set still accepts the field name ({field})",
            "" if r.returncode == 0 else r.stderr.strip().splitlines()[-1][:110],
        )

    # --- --set has to DO something -------------------------------------------
    # The two checks above pass `--set name=<its own default>`, so a CLI that
    # parses the flag and then ignores it satisfies them. That is not a
    # hypothetical shape: a generator whose values come from a frozen
    # dataclass built with no arguments accepts every flag and honours none.
    _check_set_takes_effect(doc, params, tmp)

    # --- the document has to SAY it takes the flags ---------------------------
    # A tool cannot tell from `metadata.generator.argv` which flags a CLI
    # understands. Undeclared, its only option is to write the edited values
    # into the .sdm and re-run argv as recorded, which works only if the
    # generator reads its own output; one that does not rebuilds from
    # defaults and discards the edit without anything reporting a problem.
    # A CLI that supports the flags and does not advertise them takes that
    # worse path for no reason.
    gen_now = (doc.get("metadata") or {}).get("generator") or {}
    accepts = gen_now.get("accepts") or []
    check(
        {"--out", "--set"}.issubset(set(accepts)),
        'the generator declares accepts: ["--out", "--set"]',
        ""
        if {"--out", "--set"}.issubset(set(accepts))
        else f"declares {accepts or 'nothing'} -- a tool has to fall back to "
        "staging values in the .sdm, where an edit is silently discarded "
        "unless this CEM reads its own output",
    )

    # --- ints arrive as floats ----------------------------------------------
    ints = [
        (v.get("emits") or k, v["default"]) for k, v in params.items() if v.get("type") == "int"
    ]
    if ints:
        s_i, d_i = ints[0]
        r = run(["--out", str(tmp / "c.sdm"), "--set", f"{s_i}={float(d_i)}"])
        check(
            r.returncode == 0,
            f"an int param accepts a float-encoded value ({s_i}={float(d_i)})",
            "" if r.returncode == 0 else r.stderr.strip().splitlines()[-1][:110],
        )

    # --- failures reach stderr ----------------------------------------------
    r = run(["--out", str(tmp / "d.sdm"), "--set", "definitely_not_a_param=1"])
    check(r.returncode != 0, "an unknown param is refused (non-zero exit)")
    check(
        bool(r.stderr.strip()) and not r.stdout.strip().startswith("failed"),
        "failures are written to STDERR",
        ""
        if r.stderr.strip()
        else "nothing on stderr -- a viewer shows 'authoring failed:' and no reason",
    )

    # --- the recorded generator ---------------------------------------------
    gen = doc.get("metadata", {}).get("generator") or {}
    check(bool(gen.get("argv")), "the document records a generator command")
    check(
        Path(gen.get("cwd", "/nonexistent")).exists(), "the recorded working directory still exists"
    )

    replay = list(gen.get("argv", [])[1:])
    if "--out" in replay:
        again = tmp / "replay.sdm"
        replay[replay.index("--out") + 1] = str(again)
        # Snapshot the repo around the replay. A generator that writes anything
        # besides --out corrupts the working tree on every slider release, and
        # no other check here looks at the filesystem it ran in.
        before = _tree_state()
        d_t0 = time.perf_counter()
        r = run(replay)
        d_elapsed = time.perf_counter() - d_t0
        touched = sorted(set(_tree_state().items()) - set(before.items()))
        b_ran = r.returncode == 0 and again.exists()
        check(
            b_ran,
            "the recorded command re-runs",
            "" if b_ran else r.stderr.strip().splitlines()[-1:][0][:110] if r.stderr else "",
        )
        if b_ran:
            a = {k: v["value"] for k, v in doc["params"].items()}
            b = {k: v["value"] for k, v in json.loads(again.read_text())["params"].items()}
            drift = [k for k in a if a[k] != b.get(k)]
            check(
                not drift,
                "replaying it reproduces the document EXACTLY",
                ""
                if not drift
                else f"{drift[:3]} changed -- a viewer reads that as an edit and "
                "rebuilds again, forever",
            )
            check(
                d_elapsed < SLIDER_BUDGET_S,
                f"the recorded command replays under {SLIDER_BUDGET_S:g}s",
                ""
                if d_elapsed < SLIDER_BUDGET_S
                else f"took {d_elapsed:.1f}s -- the viewer runs this BLOCKING on "
                "every slider release, so this is dead UI. Put meshing, "
                "verification and optimisation behind flags and record the "
                "generator WITHOUT them",
            )
            check(
                not touched,
                "replaying it writes nothing but --out",
                ""
                if not touched
                else f"also wrote {[k for k, _ in touched][:4]} -- a generator "
                "with side effects rewrites the repo on every slider release",
            )

    return _summary()


def _tree_state() -> dict[str, int]:
    """Path -> mtime_ns for every tracked-looking file in the CEM.

    Skips the noise a build legitimately makes (`__pycache__`, `.venv`, dot
    directories) so the check fires on real output (meshes, previews, a
    manifest) and not on a bytecode cache.
    """
    out: dict[str, int] = {}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        parts = path.relative_to(ROOT).parts
        if any(p.startswith(".") or p == "__pycache__" for p in parts):
            continue
        with contextlib.suppress(OSError):
            out[str(path.relative_to(ROOT))] = path.stat().st_mtime_ns
    return out


def _check_set_takes_effect(doc: dict, params: dict, tmp: Path) -> None:
    """Assert that some `--set` actually changes the emitted document.

    Tries params in turn rather than trusting one: nudging a single value can
    trip a legitimate cross-param constraint (a wall thicker than its own
    radius), and a build refused for a good reason is not evidence that --set
    is broken. The first param that both builds and moves is the answer; only
    if none of them does is something actually wrong.
    """
    attempted: list[str] = []
    for field, spec in params.items():
        bounds = spec.get("bounds")
        default = spec.get("default")
        if not bounds or not isinstance(default, (int, float)):
            continue
        lo, hi = float(bounds[0]), float(bounds[1])
        # A small nudge toward whichever bound has room, so the value is
        # unmistakably different without wandering into an invalid design.
        span = hi - lo
        if span <= 0:
            continue
        want = float(default) + span * 0.05
        if want > hi:
            want = float(default) - span * 0.05
        if not (lo <= want <= hi) or abs(want - float(default)) < 1e-9:
            continue
        if spec.get("type") == "int":
            want = float(round(want))
            if abs(want - float(default)) < 1e-9:
                continue
        name = spec.get("emits") or field
        attempted.append(name)
        out = tmp / f"set_{field}.sdm"
        r = run(["--out", str(out), "--set", f"{name}={want:g}"])
        if r.returncode != 0 or not out.exists():
            continue
        got = json.loads(out.read_text())["params"]
        # The document may emit under the `emits` name or the field name.
        entry = got.get(name) or got.get(field)
        if entry is None:
            continue
        if abs(float(entry.get("value", default)) - want) < 1e-6:
            check(True, f"--set actually changes the document ({name}={want:g})")
            return
        check(
            False,
            f"--set actually changes the document ({name}={want:g})",
            f"accepted the flag and emitted {entry.get('value')!r} -- the CLI "
            "parses --set and does not apply it",
        )
        return
    check(
        not attempted,
        "--set actually changes the document",
        f"tried {attempted[:4]} and none both built and moved"
        if attempted
        else "no param has usable bounds to nudge",
    )


def _summary() -> int:
    print()
    if not _fails:
        print("Contract satisfied. A tool can drive this CEM.")
        print("What this cannot tell you: whether a viewer RENDERS it. Point one at")
        print("the part and drag a slider before believing that half.")
        return 0
    print(f"{len(_fails)} check(s) failed:")
    for f in _fails:
        print(f"  - {f}")
    print("\nEach of these is invisible from inside the CEM and shows up as a")
    print("broken or unmovable control in a viewer.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
