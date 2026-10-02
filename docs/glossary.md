# Glossary

Canonical definitions for terms used across the SDM ecosystem. Source files
expand an acronym on first use in place; this page is what that expansion
links to. See [the documentation standard](documentation-standard.md) for
the rule behind that split.

## CEM (Computational Engineering Model)

A repository that describes what to build, not how to render it. A CEM may
contain natural language specifications, Python geometry definitions
written as SDF code, parameter files, material and coupling declarations,
and objective functions and constraints. It does not produce geometry
directly. It uses `sdm-core` to standardize its design into an `.sdm` file.
See [Concepts](concepts.md) for the full Define step.

## SDM (Software Defined Matter)

A design and engineering framework in which the entire path from
specification to working machine is designed and executed by software. The
output is a physical, multi-material object, manufactured directly from an
optimized design rather than from hand-authored CAD. Geometry, material
distribution, optimization targets, physical constraints, and part
couplings are all expressed as code, and that code drives a manufacturing
process directly. See [Concepts](concepts.md).

## `.sdm` file

The portable, standardized output of the Standardize step. A JSON document
containing a tree of SDF geometry nodes, named parameters with bounds and
units, material assignments, objectives, constraints, coupling
declarations, and a versioned history of design states. Any tool in the
ecosystem can read it, which makes it the lingua franca between a CEM and
the rest of the pipeline. Its schema is generated from `sdm-core`'s wire
contract rather than hand-maintained, so the schema can't drift from what
the library actually reads and writes. See [Architecture](architecture.md).

## SDF (signed distance field)

The representation SDM uses for all geometry. An SDF is a function
`f(x, y, z) → ℝ` where negative values are inside the shape, positive
values are outside, and zero is the surface. SDM writes SDFs in JAX, which
makes them differentiable by construction: gradients of a physics
objective flow back through the shape parameters. SDM never authors
geometry as a hand-built mesh; a mesh is only ever an export target, not a
design-time representation. See [Concepts](concepts.md).

## CSG (constructive solid geometry)

Building a shape by combining simple primitives, boxes, cylinders,
spheres, with boolean operations: union, subtract, intersect. SDM's
primitive catalog is composed the same way, `sdf_op` builds a CSG tree
out of `sdf_primitive` nodes, except each boolean is `min`/`max` on the
underlying distance functions rather than an edit to an explicit
boundary, so a combination that would leave degenerate topology on a
B-rep is always well-defined on an SDF. See [Concepts](concepts.md).

## TPMS (triply periodic minimal surface)

A surface that repeats periodically in three independent directions and
locally minimizes area for its boundary, the way a soap film does. The
`gyroid` primitive available in `sdm-core` is a TPMS: it produces an
internal lattice structure with continuous, self-supporting walls that
would be impractical to author by hand in CAD but is a few lines of SDF
code to express and optimize.

## SLS (selective laser sintering)

A powder bed fusion process: a laser selectively sinters successive layers
of powdered material to build a part. Because unsintered powder supports
every layer as it's built, SLS can print internal lattices, undercuts, and
articulated joints without separate support structures, which is what
makes it a target process for the compliant, gyroid-infilled geometry SDM
produces. See the manufacturing-process-agnostic principle in
[Architecture](architecture.md): SDM targets SLS as one process among
others a part's metadata can declare, not as a built-in assumption.

## MJF (multi jet fusion)

A powder bed fusion process: an inkjet array deposits fusing and detailing
agents across a powder bed layer, which infrared energy then fuses. Like
SLS, the unfused powder bed supports the part as it's built, so MJF can
produce the same class of lattice and joint geometry without added support
structures. SDM treats MJF the same way it treats SLS: a manufacturing
process a part's metadata can target, not one the format or the
optimization loop assumes.

## FEA (finite element analysis)

A numerical method that predicts how a part behaves under load by
discretizing its geometry into a mesh and solving the governing
equations over each element. SDM treats FEA as an external consumer of
the exported `.sdm` file rather than something `sdm-core` computes
in-process: `sdm-core`'s job is producing geometry and the material
assignments from `emergent-matter-sdm-materials` accurate enough for an FEA
solver downstream to trust. See [Architecture](architecture.md).

## EM (electromagnetic)

The domain of electric and magnetic field behavior, current, flux,
inductance, and the forces they produce. A design that depends on that
behavior, a motor coil, for example, declares EM material properties
(permeability, conductivity) through `emergent-matter-sdm-materials` and
hands the exported `.sdm` file to an external EM solver the same way a
structural part hands off to FEA. See [Architecture](architecture.md).

## sdm-core

The Python library that implements the ecosystem's core primitives: SDF
primitives and operations, `.sdm` file I/O, autodiff-based optimization,
constraint and manufacturing-rule validation, and export. It is the layer
every CEM depends on to move a design from definition to a standardized
`.sdm` file, and the layer that evolves that file toward manufacturability.
See [sdm-core](sdm-core.md) and the source repository,
[emergent-matter-sdm-core](https://github.com/EmergentMatter/emergent-matter-sdm-core),
for the authoritative API.

## ADR (architecture decision record)

A short document recording one decision: its status, the context that forced
it, what was decided, and the consequences that follow. Records live at
`docs/adr/NNNN-kebab-title.md` in the repository whose behaviour the decision
governs, one decision per file, and the filename states the decision rather
than only its number.

An ADR is written once and not revised. It states what was true when it was
accepted, so a later record supersedes it instead of rewriting its text. That
is what separates a record from ordinary prose: a reader can trust it as
history, while the same reasoning written into a README or a `CLAUDE.md` ages
silently, because nothing tells a reader whether it still holds. A document
that reads as a proposal is usually a decision that was made and never
recorded.

See [the documentation standard](documentation-standard.md) for where a
decision belongs relative to a docstring, a reference page, or an issue.
