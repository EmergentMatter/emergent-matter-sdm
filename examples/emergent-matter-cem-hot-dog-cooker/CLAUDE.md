# CLAUDE.md

Guidance for Claude Code when working in this example. It lives in the
`emergent-matter-sdm` docs hub as a self-contained CEM with its own
`pyproject.toml` and lock: run everything below from THIS directory, not
the hub's root. The hub's root `ruff.toml` still governs lint and format
here, and its CI runs `ruff check .` and `ruff format --check .` over it.

## What This Is

A solar hot-dog cooker: FDM-printed **square** plates carrying a
max-density grid of open tilted pads for 10.05 mm glued-on mirror tiles,
each pad molded at the exact tilt that bounces on-axis sunlight onto a
hot dog seated on a central pedestal — with per-group aim heights
optimized so the reflected power is spread evenly along the dog.
Personal/fun project of Rob's, built as an sdm-core consumer. The
Python package, the `.sdm`/STL basenames and the design's own name are
all `disco_lens` / "disco-lens", while the directory is
`emergent-matter-cem-hot-dog-cooker`; don't "fix" the package to match.

**One design lives here:** FOUR identical 205.2 mm quadrant plates
glued into a 410.4 mm square, 36×36 grid, 1284 mirrors (321/quadrant),
~90 W reflected and 84.0 W on the dog (ray-traced, CV 2.0%, spill
0.5%), aimed for the measured Ø20×136 dog. Quarter-pedestal per
quadrant + a printed clamp ring over the assembled post; one pinhole
sight tab per quadrant; no tilt screw (positioning shelved).
`DiscoLensParameters`, `build_quadrant_part` + `build_ring_part`,
`scripts/build.py`.

Two earlier designs preceded this one and are not carried here: a
single-plate version with a printed coarse tilt screw (~22 W), and a
490 mm on-edge experiment (~18 h per part). Both live in the history of
the standalone `emergent-matter-cem-hot-dog-cooker` repo, the screw at
its tag `v1`.

## Build & Run

```bash
uv sync --all-extras                         # pytest/ruff/mypy live in the dev extra
uv run pytest                                # 271 tests, ~19 s
uv run pytest tests/test_assembly.py         # one file
uv run pytest tests/test_optics.py::test_profile_flat_and_low_spill   # one test
uv run pytest -q -k "keepout or fused"       # by name
uv run mypy                                  # typecheck gate, clean
# lint + format run from the HUB root, against its ruff.toml:
#   (cd ../.. && uv run ruff check examples/emergent-matter-cem-hot-dog-cooker)
uv run python scripts/check_contract.py      # can a TOOL drive this CEM? (16 checks)
# the CEM entry point a viewer drives (--out and --set are the contract):
uv run disco-lens-build --out part.sdm [--set mirror_size=10.2] [--part ring]
#   --optimize         re-solve the aims (~1.5 s); needed when --set moves the layout
#   --validate-schema  full jsonschema pass (301 s, byte-identical result)
# the four verbs:
uv run python scripts/build.py               # optimize + write .sdm/CSV/plots (~3 s)
uv run python scripts/export_stl.py [voxel_mm]      # quadrant + ring STLs (default 0.35)
uv run python scripts/simulate_rays.py       # ray-trace the ASSEMBLED cooker
uv run python scripts/thermal_sim.py         # will it cook? 1-D transient dog model
```

sdm-core and sdm-materials install from the org package index
(`get.softwaredefinedmatter.com`, `[[tool.uv.index]]` in
`pyproject.toml`), explicit so our package names never fall through to
public PyPI. No sibling checkout is needed. To build against a local
sdm-core checkout, override for one run instead of editing the
sources: `uv run --with-editable <path to sdm-core checkout> pytest`.

## Architecture

```
src/disco_lens/
├── parameters.py    DiscoLensParameters, frozen dataclass, all dims mm
├── optics.py        the math: site grid, tilt solve, per-site stripe
│                    profiles, aim optimization, power/cook estimates
├── sdf_assembly.py  sdm-core SDF tree + .sdm Part assembly
├── sdm_ext.py       one private sdm-core extension, registered at import
├── cli.py           `disco-lens-build`: the entry point a viewer drives
├── harness.py       save + stamp metadata.generator (argv, cwd, accepts)
└── design_io.py     read solved aims back out of a mirror table
cem.toml             the DECLARED surface: every param with unit, role,
                     bounds and emitted name, plus the constraints no
                     bound can express. Read by tools WITHOUT importing
                     anything; test_manifest.py holds it to the Python
scripts/             build / export_stl / simulate_rays, plus thermal_sim.py
renders/             the README hero: scene.py (uv, mirror poses from the
                     table -> scene.json) then render.py (Blender, Cycles).
                     Presentation only, needs the STLs; nothing imports it
layout/              mirror_table.csv — one row per mirror
img/                 profile / layout / hitmap / ray-sim plots
*.sdm                geometry source of truth — BUILD ARTIFACTS, tracked in
                     git so exports/viewers work without a rebuild;
                     regenerate via the build script, never hand-edit
```

`stl/` is gitignored (meshes are regenerated); `.sdm`, `img/` and
`layout/` are tracked.

Data flow:

```
parameters → optics.design()  (grid + aim optimization, the slow part)
           → sdf_assembly.build_*_part() → .sdm  → sdm-core export → STL
           → layout/mirror_table.csv
                    ↑ simulate_rays.py and the CLI read the CSV
                      back (design_io.sites_from_table) and re-solve
                      tilts instead of re-optimizing — layout + solve
                      are deterministic, so this reproduces the .sdm
                      exactly.
```

So: **change parameters ⇒ re-run the build script first**, or the
ray-sim will silently be built from the previous design's aims.

Each script in `scripts/` is standalone: none imports another, and
anything shared lives in the package. Run them by path (`uv run python
scripts/x.py`), not as `-m`.

## Tests

`uv run pytest` — 271 tests, ~19 s. Nearly all of the wall clock is JAX
compiling the SDF closures in `test_assembly.py` and `test_sdm_ext`.
The manifest, parameter and CLI tests are most of the count and run in
about a second between them.

- `test_optics.py` — the math alone: law of reflection per site, aims
  land on the dog, profile flatness/spill, grid packing, profile
  integrates to the reflected power.
- `test_assembly.py` — compiled-SDF probes: quadrant is really one
  quadrant (seam planes), quarter-post + bore, sight tab, ring fit,
  bare pads, glass keep-outs, fused pad field, and
  `test_full_tree_matches_quadrant_in_q1`, proving the assembled tree
  agrees with the printed part. Uses cheap
  feasible aims, NOT the optimizer — it probes geometry, optics are
  `test_optics`'s job.
- `test_sdm_ext.py` — `test_fold_fast_path_matches_stock`, which
  re-proves the sector-fold shortcut against stock sdm-core on the real
  trees.
- `test_manifest.py` — `cem.toml` against the Python in both
  directions: params, defaults, bounds, emitted names, and that every
  declared constraint is enforced and every enforced one declared.
  Nothing else reads the manifest, so without this it drifts.
- `test_parameters.py` — `validate()` reports every problem at once,
  each prefixed with its constraint name, and derived values stay
  `@property`.
- `test_cli.py` — the generator stamp, the emitted-name contract, and
  failures landing on stderr. The subprocess half lives in
  `scripts/check_contract.py`; run both.

## The Math (one paragraph)

Plate aimed at the sun ⇒ incident rays are vertical in plate frame.
Mirror tilted α turns a ray by 2α: `tan 2α = (r − r_dog)/(h − z_mirror)`,
aimed at the dog's near wall. A site's stripe on the dog is the square's
meridional shadow — conv of rects `w|cos az|` and `w|sin az|` (diagonal
sites paint wider/softer) — times `cos α / sin 2α`, plus sun blur
(4.65 mrad half-angle) and glue scatter (±0.4° 1σ ⇒ stripe slides
`2ε·D/sin 2α`). Sites are binned into radius bands × checkerboard parity
(two interleaved aim groups per band — one dense band at a single height
exceeds the dog-average lineal power and can't be flattened); a
multi-start coordinate descent (with per-(group, aim) profile caching)
minimizes CV + peak-to-valley + spill of the SCATTERED profile. Result:
CV ≈ 2 % as-built, ~5 % spill, 84 W onto the dog.

## Technical Context

- **Units mm, Z-up, plate top at z = 0**, sun along −Z.
- **Open BARE pads, no pockets, no locating walls** (Rob's calls): each
  seat is just a tilted pad, mirror glued flat onto it; the pad outline
  (a 0.2 mm ledge around the mirror) is the placement guide. Locating
  stubs were removed — measured ~4% light blockage. Pitch = mirror +
  gap = 11.05 mm.
- **FUSED pad field** (Rob's call): each pad carries a wider collar
  (mirror + 0.7/side) recessed 1.5 mm below the pad plane; collars
  overlap neighbors 0.4 mm so the field prints as ONE body instead of
  thousands of per-layer islands. Mirror pitch/gap unchanged.
- **Glass keep-outs**: adjacent same-ring sites carry different
  aim-group tilts, so one pad corner can ride over the neighbor's glass
  (12 seats on the first plate printed, incl. the RIM arc at the 4
  plate corners). EVERY seat gets a keep-out cutter unconditionally — a
  model that predicted which seats needed one missed three different
  ways at three different scales — and a tilted keep-out box is
  subtracted from the WHOLE solid over every seat's mirror footprint,
  not just from the seats: at the plate corners the stiffening rim's
  arc crosses the corner seats' envelopes too.
  `test_no_solid_inside_any_glass_envelope` enforces it seat by seat.
- **sdm-core stays PRISTINE** (Rob's rule): anything this project needs
  beyond stock sdm-core lives in `src/disco_lens/sdm_ext.py`, which
  registers a single-wedge sector-fold fast path into sdm-core's
  registries at import. Never edit the checkout.
- **`validate()` returns a list, it does not raise.** An optimizer or
  an agent should see every broken rule at once, not one per round
  trip. `parameters.raise_if_invalid` is the refusing path and every
  `build_*_part` calls it, so nothing invalid reaches a `.sdm`. Each
  message is prefixed with the constraint name `cem.toml` declares;
  that prefix is what `test_manifest.py` greps for.
- **Derived values never ship in the `.sdm`.** `PARAM_FIELDS` in
  `sdf_assembly.py` names the five declared params the quadrant emits,
  with the bounds that ride into the document (a param without them is
  a control a viewer cannot move). Derived facts a reader wants, like
  `grid_pitch_mm` and `n_grid_cells`, go in `metadata` instead, where a
  second copy is a note and not a surface.
- **Requires sdm-core >= 2.0, < 3.** 2.0 changed nothing this repo
  calls; the floor is there because the tracked `.sdm` files are built
  with it (it stopped writing an empty `couplings` record). What 0.4
  changed for us still holds: `canonical_sector_fold` is jit-safe
  upstream now, so `sdm_ext` no longer carries that patch; the fold also gained two
  neighbour-wedge evaluations, so a folded child that spills past its
  wedge no longer over-reports distance. `sdm_ext` still skips those two
  evaluations, purely for speed (stock costs 2.9x on the seat field),
  and `test_fold_fast_path_matches_stock` proves the shortcut changes
  nothing for THIS plate: the pads sit strictly inside their wedge, so
  the nearest copy to any point is its own. If a pad ever crosses a
  seam, that test fails and the fast path goes, not the test.
- **Quadrant trick:** the tree is authored as the FULL plate with
  4-fold symmetry, so the printed part is literally that tree
  intersected with the x≥0, y≥0 box (`build_tree(b_quadrant=)`)
  — quarter-pedestal, seam faces, and per-quadrant sight tab fall out
  for free. The even grid keeps every pad 0.3 mm clear of the seam
  planes. `b_quadrant=False` gives the assembled cooker for ray tracing.
- **ONE tree variant.** Every builder authors the folded tree: one
  quadrant of seats under `canonical_sector_fold`, cheap to evaluate.
- **4-fold fold:** the centered even grid is exactly 4-fold symmetric;
  only quadrant-1 seats are authored, wrapped in one
  `canonical_sector_fold(n=4)`. **Gotcha: a 90° rotation of an even
  checkerboard FLIPS parity**, so aim-group parity is assigned from each
  site's canonical Q1 representative (`layout_sites`) — otherwise the
  built plate disagrees with the optics model (this bit us; tests catch
  it now).
- **Pad tilt is a conjugation** `Rz(az)·Ry(−α)·Rz(−az)`: pads lean toward
  the center while their edges stay along the plate axes (grid packing
  needs axis-aligned squares; a flat mirror doesn't care about edge
  orientation, only its normal).
- **Transform signs:** sdm-core transforms act on the query point —
  `translate(t)` moves geometry +t but `rotate_*(angle=a)` rotates
  geometry −a. Wrapped once in `sdf_assembly._rot_*_geometry`; verified
  numerically by `tests/test_assembly.py` probing pads of folded
  copies.
- **Pad rooting:** pads root 2.5 mm into the 3 mm slab; steep sites poke
  below the underside and a box cutter trims the bottom flat.
- **Optimization is deterministic** (seeded jitter starts) and cached
  per (group, aim height). It is also FAST: the aim solve is ~1.4 s
  for 1284 sites. The "~15 min build" this repo believed in for a long
  time was never the optimizer, it was the jsonschema pass inside
  `save` (301 s on the 2.1 MB quadrant, for a byte-identical file).
  `harness.save_verified` writes without it and loads the file back
  instead, so `build.py` now finishes in ~3 s. Tests rely on the
  reproducibility, which is verified: a rebuild reproduces the mirror
  table, the ring document and every plot byte for byte, and the
  quadrant document apart from its date stamp.
- **Marching-cubes pinholes:** shallow-graze faces can leave a few
  non-manifold fins at fine voxel sizes. `scripts/export_stl.py`
  strips duplicate/degenerate faces, fills holes, and reports
  watertightness.

## STL hygiene (Rob's rule)

`stl/` holds ONLY the current version of each part; every export moves
older timestamps of that part into `stl/archive/` automatically
(`archive_older` in `scripts/export_stl.py`). Nothing is deleted.
Reason: a mesh review tool lists every STL in the folder, and a stale
timestamp in that dropdown means marking up last week's geometry.

## Naming Conventions

Hungarian prefixes: `d_` float, `n_` int, `b_` bool, `s_` string;
functions named by what they return. sdm-core boundary uses its plain
snake_case API.

## Known Issues / Future Work

- **No upstream candidates outstanding for THIS design.** The repo is
  a pure stock sdm-core consumer now: no vendored patches, no private
  primitives, just the one performance fast path in `sdm_ext.py`. The
  helix/thread gaps found while building the v1 screw are still open
  upstream; the write-up and patches are at tag `v1` of the standalone
  repo.
- **No release control of its own.** As an example in the docs hub it
  has no version, changelog or CI; the hub's changelog and gates cover
  it.
- Cook model is lumped-capacitance with a guessed film coefficient and a
  100 °C surface cap; treat "core hot in 3–4 min" as order-of-magnitude.
- Mirror thickness (2 mm) is assumed, not measured — the optics stack
  height uses it; check the actual tiles.
- The skewer is a round 4 mm steel skewer by assumption; hub hole Ø4.1,
  printed slightly undersize — ream to fit. Flat skewers need a
  rectangular hole instead.
