"""The CLI contract, in-process.

These cover the parts a unit test can see. The parts it cannot are in
``scripts/check_contract.py``, which drives the CLI as a SUBPROCESS the
way a tool does: an in-process call shares ``sys.argv``, swallows
nothing, and hides exactly the stream and exit-code bugs that matter.
Run both.
"""

from __future__ import annotations

import json

import pytest

from disco_lens.cli import main
from disco_lens.harness import NON_REPLAYABLE_FLAGS
from disco_lens.parameters import DiscoLensParameters
from disco_lens.sdf_assembly import PARAM_FIELDS


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One ring build, reused: the ring needs no mirror table."""
    out = tmp_path_factory.mktemp("cli") / "ring.sdm"
    argv = ["--out", str(out), "--part", "ring"]
    assert main(argv) == 0
    return out, json.loads(out.read_text())


def test_it_builds(built):
    out, doc = built
    assert out.exists()
    assert doc["materials"]


def test_the_generator_is_recorded(built):
    """Without it the document can be rendered and never re-derived."""
    _, doc = built
    gen = doc["metadata"]["generator"]
    assert gen["argv"]
    assert gen["cwd"]


def test_the_generator_declares_the_flags_it_takes(built):
    """A tool cannot tell which flags a CLI understands from argv."""
    _, doc = built
    assert {"--out", "--set"} <= set(doc["metadata"]["generator"]["accepts"])


def test_the_flag_that_changes_nothing_is_not_recorded(tmp_path):
    """--validate-schema costs 301 s on the quadrant and produces a
    byte-identical file, so replaying it spends a viewer's whole budget
    to change nothing."""
    assert "--validate-schema" in NON_REPLAYABLE_FLAGS
    out = tmp_path / "ring.sdm"
    assert main(["--out", str(out), "--part", "ring", "--validate-schema"]) == 0
    argv = json.loads(out.read_text())["metadata"]["generator"]["argv"]
    assert "--validate-schema" not in argv


def test_the_flag_that_replay_needs_is_recorded(tmp_path):
    """--optimize is cheap (1.4 s) and deterministic, and a document
    built with a layout-changing --set can ONLY be rebuilt by
    re-solving. Stripping it would record a command that exits 2."""
    assert "--optimize" not in NON_REPLAYABLE_FLAGS
    out = tmp_path / "small.sdm"
    assert main(["--out", str(out), "--set=d_plate_size=300.0", "--optimize"]) == 0
    argv = json.loads(out.read_text())["metadata"]["generator"]["argv"]
    assert "--optimize" in argv


@pytest.mark.parametrize("entry", PARAM_FIELDS, ids=lambda e: e[1])
def test_set_accepts_the_emitted_name(tmp_path, entry):
    """A viewer reads the EMITTED name and hands that back on re-emit,
    so a CLI that only knows field names refuses every rebuild it
    sends. d_plate_size ships as assembled_plate_size, which prefix
    stripping alone does not produce."""
    s_field, s_emits, _ = entry
    d_default = getattr(DiscoLensParameters(), s_field)
    out = tmp_path / f"{s_emits}.sdm"
    assert main(["--out", str(out), "--part", "ring", f"--set={s_emits}={d_default}"]) == 0


def test_set_by_field_name_changes_the_document(tmp_path):
    out = tmp_path / "a.sdm"
    assert main(["--out", str(out), "--set=d_base_thickness=3.4", "--part", "ring"]) == 0
    assert main(["--out", str(tmp_path / "b.sdm"), "--part", "ring"]) == 0
    assert out.read_text() != (tmp_path / "b.sdm").read_text()


def test_an_unknown_param_is_refused(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(["--out", str(tmp_path / "x.sdm"), "--set=not_a_param=1"])
    assert e.value.code != 0


def test_a_near_miss_is_named(tmp_path, capsys):
    """Naming the near miss turns a typo into a one-line fix."""
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "x.sdm"), "--set=n_base_thickness=3"])
    assert "d_base_thickness" in capsys.readouterr().err


def test_invalid_parameters_fail_on_stderr(tmp_path, capsys):
    """A viewer's re-emit reads stderr to explain a failure and discards
    stdout, so the reason has to be on the right stream."""
    code = main(
        ["--out", str(tmp_path / "x.sdm"), "--set=d_sight_hole_diameter=9.0", "--part", "ring"]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "sight_bore_leaves_boss_wall" in captured.err
    assert captured.out == ""
    assert not (tmp_path / "x.sdm").exists()


def test_a_table_that_does_not_cover_the_design_is_refused(tmp_path, capsys):
    """Reusing solved aims is only sound while the table describes THIS
    layout. A smaller plate lays out fewer sites, so the lookup has to
    fail loudly rather than half-apply."""
    code = main(["--out", str(tmp_path / "x.sdm"), "--set=d_plate_size=300.0"])
    captured = capsys.readouterr()
    assert code == 2
    assert "--optimize" in captured.err
