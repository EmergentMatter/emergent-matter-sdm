#!/usr/bin/env bash
# Run the whole FEA workflow on the current backpack_table_hook.sdm.
#
#   ./run_fea.sh            # everything (~30 min on an M-series laptop; the 3-D step is ~22 min)
#   FEA_SKIP_3D=1 ./run_fea.sh   # 2-D only (~5 min); reuses the last 3-D knockdown
#
# Runs in this example's own uv environment (pyproject.toml), which has
# sdm-core and jax-fem side by side.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p fea/logs

fea() {  # fea <log-name> <script> : run a script in the example's environment
  local s_log="fea/logs/$1.log"; shift
  echo "== $* (log: $s_log)"
  PYTHONUNBUFFERED=1 uv run --locked python -B "$@" 2>&1 \
    | grep --line-buffered -v '\[DEBUG\]\|\[INFO\]' | tee "$s_log"
}

fea verify    fea/00_verify_elements.py                 # solver sanity check on a cantilever
fea 2d        fea/01_fea_2d.py                          # production run, 0.25 mm walls
FEA_H_FINE=0.5 fea 2d_h0.5 fea/01_fea_2d.py             # mesh convergence (half resolution)
FEA_H_FINE=2.0 fea 2d_h2   fea/01_fea_2d.py             # coarse twin of the 3-D mesh
if [[ "${FEA_SKIP_3D:-0}" != 1 ]]; then
  rm -f fea/results_3d_extruded.json
  fea 3d      fea/02_fea_3d.py                         # handle-width + layer-adhesion check
fi
fea render2d  fea/03_render_2d.py
fea render3d  fea/04_render_3d.py
fea summary   fea/05_summary.py                         # safety factors -> fea/summary.json
