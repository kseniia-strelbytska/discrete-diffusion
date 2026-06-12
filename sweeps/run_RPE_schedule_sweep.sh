#!/usr/bin/env bash
# Sequential sweep: RPE architecture across schedules and T values.
# Estimated runtime: 5 × ~1.5h = ~7.5h total.
#
# Usage:
#   bash sweeps/run_RPE_schedule_sweep.sh          # run all 5
#   bash sweeps/run_RPE_schedule_sweep.sh 3        # resume from run 3

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

START_FROM="${1:-1}"

run() {
    local run_num="$1"
    local config="$2"
    local label="$3"

    if [ "$run_num" -lt "$START_FROM" ]; then
        echo "=== Skipping run ${run_num}/5: ${label} ==="
        return
    fi

    echo ""
    echo "========================================="
    echo "  Run ${run_num}/5: ${label}"
    echo "  Config: ${config}"
    echo "  Started: $(date)"
    echo "========================================="

    python src/main.py --config "${config}" --save

    echo "  Finished: $(date)"
}

run 1 config_sweep1_RPE_uniform_T100.yaml     "RPE + uniform,       T=100"
run 2 config_sweep2_RPE_gaussian_s20_T100.yaml "RPE + Gaussian σ=20, T=100"
run 3 config_sweep3_RPE_gaussian_s10_T100.yaml "RPE + Gaussian σ=10, T=100"
run 4 config_sweep4_RPE_gaussian_s5_T100.yaml  "RPE + Gaussian σ=5,  T=100"
run 5 config_sweep5_RPE_uniform_T500.yaml      "RPE + uniform,       T=500"

echo ""
echo "========================================="
echo "  All 5 runs complete: $(date)"
echo "========================================="
