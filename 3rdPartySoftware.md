# Third-Party Software

Every software dependency of `emergent-matter-sdm` that is **not** EmergentMatter first-party code: the packages resolved into this repository's environment, the build backend, and the third-party GitHub Actions its CI runs.

Versions are deliberately left out. They move on every lock refresh, and a list that churns on every bump stops being read. For the exact pinned version of anything below, see `uv.lock`.

## Direct dependencies

Chosen by this repository and declared in `pyproject.toml`.

### Runtime

`[project] dependencies`, installed for anyone who installs this package.

| Package | License | Purpose |
|---|---|---|
| [`numpy`](https://numpy.org) | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | Fundamental package for array computing in Python |

### Optional extras

`[project.optional-dependencies]`, installed only when the extra is requested.

_None._

### Development

`[dependency-groups]`, used for tests, linting and type checking. Not shipped to consumers.

| Package | License | Purpose |
|---|---|---|
| [`mypy`](https://www.mypy-lang.org/) | MIT | Optional static typing for Python |
| [`pytest`](https://docs.pytest.org/en/latest/) | MIT | pytest: simple powerful testing with Python |
| [`ruff`](https://docs.astral.sh/ruff) | MIT | An extremely fast Python linter and code formatter, written in Rust |

## Transitive dependencies

Not requested by this repository. They arrive as dependencies of the packages above and are resolved into `uv.lock`.

| Package | License | Purpose |
|---|---|---|
| [`ast-serialize`](https://github.com/mypyc/ast_serialize) | MIT | Python bindings for mypy AST serialization |
| [`colorama`](https://github.com/tartley/colorama) | BSD | Cross-platform colored terminal text |
| [`iniconfig`](https://github.com/pytest-dev/iniconfig) | MIT | brain-dead simple config-ini parsing |
| [`librt`](https://github.com/mypyc/librt) | MIT | Mypyc runtime library |
| [`mypy-extensions`](https://github.com/python/mypy_extensions) | MIT | Experimental type system extensions for mypy |
| [`packaging`](https://github.com/pypa/packaging) | Apache-2.0 OR BSD-2-Clause | Core utilities for Python packages |
| [`pathspec`](https://github.com/cpburnz/python-pathspec) | MPL-2.0 | Utility library for gitignore style pattern matching of file paths |
| [`pluggy`](https://pypi.org/project/pluggy/) | MIT | plugin and hook calling mechanisms for python |
| [`pygments`](https://pygments.org) | BSD-2-Clause | Pygments is a syntax highlighting package written in Python |
| [`typing-extensions`](https://github.com/python/typing_extensions) | PSF-2.0 | Backported and Experimental Type Hints for Python 3.9+ |

## Build and CI toolchain

Third-party software the repository is built and tested with, rather than packages it imports.

| Component | Role | License |
|---|---|---|
| [Python](https://www.python.org/) | Language runtime | PSF-2.0 |
| [`uv`](https://github.com/astral-sh/uv) | Dependency resolver and installer | MIT OR Apache-2.0 |
| [`hatchling`](https://github.com/pypa/hatch) | Build backend, `[build-system] requires` | MIT |
| [`actions/checkout`](https://github.com/actions/checkout) | CI: checks out the repository | MIT |
| [`actions/upload-artifact`](https://github.com/actions/upload-artifact) | CI: uploads build artifacts | MIT |
| [`astral-sh/setup-uv`](https://github.com/astral-sh/setup-uv) | CI: installs `uv` | MIT |

## First-party, deliberately not listed

These are EmergentMatter's own code and are out of scope for this file:

- Any `emergent-matter-*` package, and `sdm-view`
- `EmergentMatter/actions`, the shared release and changelog workflows

## Notes

- `mypy-extensions` publishes no license metadata to PyPI. MIT is taken from the `LICENSE` file in its upstream repository.
- `pathspec` is MPL-2.0, which is file-level copyleft. Unmodified transitive dependency, so the obligation is to keep the notices intact, which distributing them unmodified does.
- No GPL or LGPL licensed package appears in this repository's tree.
- The `setup-node` and `setup-bun` steps in `.github/workflows/` sit inside commented-out template blocks, and this repository has no `package.json`, so it has no JavaScript dependencies.

## Regenerating

Direct entries come from `pyproject.toml`; transitive entries are every remaining package in `uv.lock` that is not first-party. Licenses come from installed package metadata, falling back to PyPI.
