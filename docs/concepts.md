# SDM Concepts

## What is Software Defined Matter?

Software Defined Matter (SDM) is a design and engineering framework where the entire path from specification to working machine is designed and executed by software. The final output is a physical, multi-material object, manufactured directly from the optimized design.

The core claim: geometry, material distribution, optimization targets, physical constraints, and part couplings can all be expressed as code, and that code can be used to drive a manufacturing process directly, without human-in-the-loop CAD authoring.

## Define → Standardize → Evolve

```
CEM  →  .sdm file  →  Evolved .sdm  →  Manufactured part
```

### Step 1: Define a CEM

A **Computational Engineering Model (CEM)** is a repository that describes what you want to build. It may contain:

- Natural language specifications
- Python geometry definitions (SDF code)
- JSON parameter files
- Material and coupling declarations
- Objective functions and constraints

The CEM is the engineering intent. It does not produce geometry directly; it produces an `.sdm` file.

### Step 2: Standardize to `.sdm`

The CEM uses the `sdm-core` library to serialize its design into an `.sdm` file. This file is:

- **Portable**: can be passed between machines, services, and tools
- **Versioned**: snapshots of design state
- **Consumable**: any SDM-ecosystem tool can read it

The `.sdm` file contains: shapes (SDF trees), parameters, materials, objectives, constraints, and coupling declarations.

Parameters survive the trip. A dimension driven by a `Param` is written as a `$ref` to that param, not as the number it currently resolves to, so a tool that loads the file sees `{'r': {'$ref': 'radius'}}` rather than `{'r': 8.0}` and can still vary it. The round trip is the point of the format: an `.sdm` file is meant to move between tools, not to sit beside the script that made it.

### Step 3: Evolve

`sdm-core` operates on the `.sdm` file to move the design toward manufacturability:

- **Optimize**: autodiff-based cost minimization
- **Validate**: check physical constraints
- **Integrate**: assemble multiple parts via coupling declarations
- **Export**: produce geometry for the target manufacturing process

## Key abstractions

### Signed Distance Fields (SDFs)

Geometry in SDM is always expressed as a **signed distance field**: a function `f(x,y,z) → ℝ` where negative values are inside the shape and positive values are outside. SDFs are written in JAX, making them differentiable by construction.

Primitives (`capped_cylinder`, `gyroid`, `notch_hinge`, …) are combined via boolean operations (`union`, `subtract`, `intersect`) and spatial transforms (`translate`, `rotate`, `scale`).

### Parameters

Geometry is **parametric**. Dimensions are named parameters (`make_param_ref("height")`) rather than literals. The optimizer varies these parameters to satisfy objectives and constraints.

### Materials

Parts declare which materials they may use. The material substrate (`emergent-matter-sdm-materials`) provides multi-physics properties (structural, electromagnetic, thermal, manufacturing) with provenance tracking.

### Couplings

Parts declare how they connect to other parts: mechanical ports, hinge axes, magnetic interfaces, etc. Couplings allow assembly integration without hand-tuning interface geometry.
