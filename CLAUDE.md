# emergent-matter-sdm: Claude Code Instructions

This is the **documentation and getting-started hub** for the Software Defined Matter (SDM) ecosystem.

## Code standard

Follow [STYLE.md](STYLE.md) at the repo root for all code, comments, tests,
and docs. It is the org standard and wins over habit. Pull request
descriptions follow
[.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md).
For where a piece of documentation belongs, not how to write it, see
[docs/documentation-standard.md](docs/documentation-standard.md).

## What this repo is

- The open-source landing page for SDM and `sdm-core`
- Conceptual documentation: what SDM is, how it works, the pipeline
- Cross-repo ecosystem overview

It is **not** a code library. The only code here is `src/sdm_showcase/`, the
renderer that regenerates the primitives showcase board embedded in the
README; it consumes `sdm-core`, it does not implement it. The authoritative
implementation lives in `emergent-matter-sdm-core`.

## Repo layout

```
docs/              Core documentation pages
  concepts.md      What SDM is: CEM, .sdm files, the pipeline
  sdm-core.md      sdm-core library reference / overview
  ecosystem.md     All SDM repos and how they fit together
  architecture.md  System design, data flow, .sdm file format
  img/             Renders and logo used by README and docs pages
src/sdm_showcase/  Renderer for the primitives showcase board (needs sdm-core + Blender)
```

## Writing style

- Documentation targets a technically sophisticated reader (ML/robotics engineer) who is new to SDM
- Lead with concrete examples, then explain the concept
- Use code blocks liberally: show the actual Python API
- Mermaid diagrams for data flow and architecture
- Avoid marketing language; be precise and honest about what exists vs. what is planned

## Key concepts to keep consistent

- **CEM**: Computational Engineering Model, an external repo that describes a part/system
- **.sdm file**: the portable standardized output (JSON-based); the lingua franca of the ecosystem
- **sdm-core**: the Python library; provides primitives, SDF ops, file I/O, optimization, export
- **SDF**: signed distance field; geometry is always expressed as JAX SDF, never as hand-authored mesh
- The pipeline order is fixed: **Define → Standardize → Evolve**

## When adding or editing docs

Before editing, check `emergent-matter-sdm-core/README.md` for the authoritative description of any sdm-core API or concept. The two repos should agree; if they diverge, trust the sdm-core source and update this repo.

Read the open issues and the project board before starting work; plans live there now, not in this file.
