# The EmergentMatter style guide

How to write, test, and document code in EmergentMatter repos.

- [Python style](#python-style)
- [Naming](#naming)
- [Docstrings and comments](#docstrings-and-comments)
- [Typing](#typing)
- [Python version policy](#python-version-policy)
- [Testing](#testing)
- [TypeScript](#typescript)
- [Documentation](#documentation)
- [Open-source hygiene](#open-source-hygiene)

## Python style

- **Use `uv` only.** `uv sync`, `uv run pytest`, `uv run ruff check .`. No
  bare `pip`, `poetry`, or `python -m venv`.
- **Use src-layout + hatchling.** `src/<package>/`, not a top-level
  package directory.
- **Commit `uv.lock`.** Never gitignore it. CI syncs with `uv sync
  --locked`, so a stale lockfile fails loudly.
- **Ruff owns linting and formatting.** Every repo carries a base
  ruleset file and a local override file:
  - `ruff-base.toml`: the shared ruleset. Do not edit it. Put local
    changes in `ruff.toml` instead.
  - `ruff.toml`: `extend = "ruff-base.toml"` plus this repo's own
    additions.
  - Line length is 100. Target Python is `py313`; a repo on the 3.11
    floor sets `target-version = "py311"` in its own `ruff.toml`.
  - Put every mechanical reformat commit in `.git-blame-ignore-revs`.
  - Opt a catalog or data module with hand-aligned literals out per-file
    with `# fmt: off` / `# fmt: on`. Never override `[format]` for the
    whole repo.
- **Add `from __future__ import annotations`** at the top of every
  module. Exception: a Blender `PropertyGroup` module, which needs
  evaluated annotations at class-definition time. State the reason in a
  comment.
- **Import heavy or host-bound dependencies lazily** (`jax`, `bpy`,
  `openvdb`, and similar): import at the point of use, not at module load
  time. Raise a clear error that tells the user how to get the dependency
  if the import fails.
- **Use frozen dataclasses with `__post_init__` validation** for schemas.
  Validate at the boundary. Name the class, the field, the expected
  value, and the actual value in the error message. State a reason in
  review for a new mutable dataclass in schema code.
- **Write CLI scripts as** `argparse`, `def main(argv: list[str] | None =
  None) -> int`, errors to stderr with an `error:` prefix, `return 1` on
  anything ambiguous. Never a silent skip.

## Naming

The typed-prefix convention below applies to engineering quantities:
geometry, physics, catalog data. Generic tooling and infrastructure code
uses plain descriptive names instead.

- **Use the typed-prefix convention on primitive scalars**: `d_` for
  `float`, `n_` for `int`, `b_` for `bool`, `s_` for `str`. It keeps a
  variable's type visible at every call site. Applies to scalars only;
  give a composite value (a vector, a dataclass, a mapping) a bare name
  plus a type annotation instead.
- **Put units in the name** for a scalar accessor: `d_min_wall_mm`, not
  `d_min_wall`.
- Use `SCREAMING_SNAKE` for module-level constants.
- Use `_leading_underscore` for private names. Never put them in
  `__all__`.
- Declare `__all__` in every public module.
- End exception names in `Error`, `PascalCase`.

## Docstrings and comments

Write rich docs at module and API boundaries. Keep function bodies lean.

1. **Open every module with a docstring that says why it exists**, and
   where it sits relative to the rest of the package.
2. **Give every public function or class a docstring.** One line when
   the signature is self-explanatory. Google-style `Args:` / `Returns:`
   / `Raises:` sections when it isn't.
3. **Write inline comments that explain why**: invariants, units,
   coordinate frames, a rejected alternative and the reason it lost.
   Never restate what the next line does. Delete a comment that just
   restates the code.
4. **Give private helpers a docstring only when they're non-obvious.** A
   one-line `d_*` property whose name already encodes quantity and unit
   needs nothing further.

Keep out of code:

- **Migration history.** Put "moved here from X", "this used to work
  differently" in the changelog or the commit body, not in a docstring
  or a loop-body comment.
- **No `TODO` / `FIXME` / `HACK` markers.** File a GitHub issue, raise
  `NotImplementedError` with a message, or write a "Known Limitations"
  section instead.
- **Give every `noqa` a rule code and a one-line justification.**
  `# noqa: E501` alone is not acceptable.

## Typing

- **Ship `py.typed`** in every package.
- **Run mypy in CI.** Use `disallow_untyped_defs` and modern syntax by
  default. Relax per-module for JAX-kernel or `bpy`-bound code with an
  explicit, commented override in `[[tool.mypy.overrides]]` -- never a
  blanket project-wide relaxation. Set `ignore_missing_imports` for
  `jax`, `pyvista`, `trimesh`, `skimage`, `bpy` as needed.
- **Generate a `Literal`'s runtime counterpart with `typing.get_args()`.**
  Never hand-copy a tuple that can drift from it.
- **Return multi-value results as a `NamedTuple` or dataclass.** Never a
  bare positional tuple.

## Python version policy

- Default to `requires-python = ">=3.13"`.
- A library with zero runtime dependencies, or stdlib-only ones, may
  declare `>=3.11` -- only paired with a CI matrix that runs its tests on
  3.11.

## Testing

- **Name tests as behavior sentences**:
  `test_removed_workflow_path_is_broken_not_merely_stale`, not
  `test_check_3`.
- **Write module-level test functions, no test classes.** Use a flat
  `tests/` directory, with files named for the module or the invariant
  they pin.
- **Open every test-module docstring by stating the invariant the file
  pins.**
- **Prefer helpers over fixtures.** Use module-level builder functions
  (`_make_header(**overrides)`) plus the built-in `tmp_path` and
  `monkeypatch`. Add a `conftest.py` only when data is genuinely shared
  across files, with a comment explaining why.
- **Parametrize once cases exceed about three**, with `ids=`.
- **Pin the message in negative tests**: `pytest.raises(SomeError,
  match="...")`.
- **Use `pytest.approx` with an explicit, justified tolerance** for
  numeric asserts. Name suite-wide tolerances as constants
  (`RATE_TOL = 1.02  # margin absorbs sampling luck, not real error`).
  Never use bare `==` on a computed float.
- **Write a happy-path test and an edge-case test for every public API.
  Add a regression test with every bugfix.** Name gate tests
  (provenance, units, a cross-file contract) as gates.
- **Report coverage; don't gate on it by default.** A repo may opt into
  a coverage gate.
- **Read machine-specific paths from environment variables, with a loud
  skip when unset.** Never hardcode an absolute path.
- **Don't parallelize a fast suite.** `pytest -n auto` and compiler
  caches are for a suite that has become the slow part of CI; below about
  30s they cost more than they save. When a suite does get there, measure
  before and after -- the CI template's test job explains both levers.

## TypeScript

For repos with a TypeScript package:

- **Use Bun as the package manager and script runner.** Run every tool
  through `bun run <script>`. Declare the bundler and test runner in
  `package.json`'s scripts; never invoke them by another route.
- **Commit the text-format Bun lockfile.** Never commit a second
  lockfile alongside it.
- **Pin the toolchain.** Record the Bun version in `package.json`'s
  `engines` field, and pin the same version in CI. Never install Bun
  from an unversioned script in CI.
- **Install with `bun install --frozen-lockfile`** in CI. Give any local
  script that installs as a side effect the same flag, so a launch can't
  mutate the lockfile.
- **Write runner-agnostic tests.** Import test primitives from the
  package's harness module, never from the runner package directly.
  Never use `it.each`; write a `for` loop over a table instead.
- **Run `format:check` in CI, never `format`.** `format` mutates files.
- **Let a cross-language or parity test skip locally when a runtime is
  missing.** In CI, install every runtime the suite needs so those tests
  run instead of skipping. A skip in CI is a misconfiguration, not a
  pass.
- **Name the sanctioned toolchain in error and skip messages.** A
  message telling someone how to regenerate a file or install a
  dependency names the current package manager, never a retired one.
- **Commit generated contract files** (an OpenAPI-generated `schema.ts`,
  for example), so their diff shows contract changes in review.
- **Set `tsconfig` to strict, plus**: `noUncheckedIndexedAccess`,
  `exactOptionalPropertyTypes`, `noImplicitOverride`,
  `verbatimModuleSyntax`.
- **Use ESLint flat config plus Prettier.** Set `no-explicit-any` to
  error. No empty `catch` blocks.
- **Write function components with named exports only.**
- **Keep pure logic in React-free modules**, so it's unit-testable
  without a DOM.

## Documentation

- **Give the README a working Install section and a copy-pasteable
  quickstart that runs from a clean clone.** Keep version numbers, test
  counts, and layout trees accurate at release time.
- **Keep `CLAUDE.md` to the stable guide**: build/run commands,
  architecture, naming, constraints. Put dated journals, roadmaps, and
  sign-off notes in GitHub issues or `docs/journal/`.
- **Write ADRs (architecture decision records)** at
  `docs/adr/NNNN-kebab-title.md`, with `Status`, `Context`, `Decision`, and
  `Consequences` sections. A record states what was true when it was
  accepted, so it is superseded by a later one rather than rewritten.
- **For where a piece of documentation belongs**, not how to write it, see
  [`documentation-standard.md`](https://github.com/EmergentMatter/emergent-matter-sdm/blob/main/docs/documentation-standard.md)
  in the docs hub.

### Writing

- **Use plain punctuation.** No em dashes.
- **Write short sentences.** No run-ons.
- **Never count or place things in docs or comments.** A phrase like
  "the six files," "files 4-9," or "all three jobs" goes stale the
  moment the set changes. Name things by role or filename instead. A
  configuration value (a line length, a version number, a timeout) is
  not a count of things, and stating it directly is fine.
- **Expand an acronym on first use in a file**, then link the
  [glossary](https://github.com/EmergentMatter/emergent-matter-sdm/blob/main/docs/glossary.md)
  rather than redefining the term. The glossary in the docs hub holds
  the canonical entry; a second definition somewhere else is the one
  that goes stale.

## Open-source hygiene

Checklist before a repo goes public, or during PR review on one that
already is. **These rules apply to every file that ships in the repo**
-- `CLAUDE.md`, docs, test fixtures, scripts, comments, error messages,
not only source code:

- **Never name a person as decision provenance** in shipped source or
  docs. Use an institutional source string plus a dated entry in a
  private decision log.
- **Never hardcode an absolute personal path.** Derive it from
  `__file__`, or read an environment variable with a loud skip when it's
  unset.
- **Never reference a private repo as resolvable.** Describe the pattern
  in place, or link a public doc.
- **Never cite an internal tracker or doc** in a user-facing string. Use
  a public issue, an in-repo doc, or plain prose.
- **Remove internal process artifacts**: agent-facing journals, "note to
  future me" comments, bootstrap prompt transcripts. Delete them.
- **Complete `[project]` metadata**: institutional `authors`, `urls`,
  `readme`, `classifiers`.
- **Make cross-repo installs work from a public clone.** Publish a
  sibling dependency pinned as a local path (`../other-repo`) to an
  index or a wheelhouse before the repo goes public.
