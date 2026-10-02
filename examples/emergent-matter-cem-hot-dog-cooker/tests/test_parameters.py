"""The parameter contract: derived values, and how validation reports.

``validate`` returning a list rather than raising is the part worth
testing. An optimizer or an agent that gets one problem per round trip
fixes five things in five passes, and the fifth pass is where someone
gives up and relaxes a constraint instead.
"""

from __future__ import annotations

import dataclasses

import pytest

from disco_lens.parameters import DiscoLensParameters, raise_if_invalid

ALL = [DiscoLensParameters]


@pytest.mark.parametrize("cls", ALL)
def test_defaults_are_valid(cls):
    """Every shipped design has to pass its own rules."""
    assert cls().validate() == []


@pytest.mark.parametrize("cls", ALL)
def test_validate_returns_a_list_and_does_not_raise(cls):
    """Even a thoroughly broken design comes back as data."""
    p = dataclasses.replace(cls(), d_sight_hole_diameter=9.0, d_stick_hole_diameter=30.0)
    problems = p.validate()
    assert isinstance(problems, list)
    assert all(isinstance(s, str) for s in problems)


def test_all_problems_are_reported_at_once():
    """The whole point: not one problem per round trip."""
    p = dataclasses.replace(
        DiscoLensParameters(), d_sight_hole_diameter=9.0, d_stick_hole_diameter=30.0
    )
    problems = p.validate()
    assert len(problems) >= 3, problems


@pytest.mark.parametrize("cls", ALL)
def test_every_problem_is_prefixed_with_its_constraint_name(cls):
    """cem.toml joins to validate() by name, so the name has to be in
    the message. test_manifest.py greps the source for these."""
    p = dataclasses.replace(cls(), d_sight_hole_diameter=9.0, d_stick_hole_diameter=30.0)
    for s_problem in p.validate():
        s_name = s_problem.split(":")[0]
        assert s_name.replace("_", "").isalnum(), s_problem
        assert s_name.islower(), s_problem
        assert ": " in s_problem, s_problem


def test_raise_if_invalid_raises_and_names_everything():
    """build_*_part is what refuses, and it refuses with the full list."""
    p = dataclasses.replace(
        DiscoLensParameters(), d_sight_hole_diameter=9.0, d_stick_hole_diameter=30.0
    )
    with pytest.raises(ValueError) as e:
        raise_if_invalid(p)
    for s_problem in p.validate():
        assert s_problem in str(e.value)


@pytest.mark.parametrize("cls", ALL)
def test_raise_if_invalid_passes_a_good_design(cls):
    raise_if_invalid(cls())


@pytest.mark.parametrize("cls", ALL)
def test_derived_values_are_properties_not_fields(cls):
    """A derived value written down as a field is a second source of
    truth that nothing keeps in sync, which is how a param edit produces
    geometry that violates the model with nothing reporting a problem.
    """
    s_fields = {f.name for f in dataclasses.fields(cls)}
    derived = {n for n in dir(cls) if isinstance(getattr(cls, n, None), property)}
    assert not (s_fields & derived), s_fields & derived
