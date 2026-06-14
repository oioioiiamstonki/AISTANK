#!/usr/bin/env bash
# Two-phase soccer pipeline, run detached by a persistent wsl.exe.
#   Phase 1: imitation/locomotion  -> mjx/checkpoints/walk_fwd.pkl
#   Phase 2: self-play soccer (warm-started from Phase 1) -> mjx/checkpoints/soccer.pkl
# Each phase auto-resumes from its checkpoint if it ever crashes.
#
#   wsl -d Ubuntu-24.04 -- bash mjx/run_pipeline.sh [PHASE1_HOURS] [PHASE2_HOURS]
set -u
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
P1H="${1:-2}"
P2H="${2:-10}"
echo "=== PIPELINE start $(date): phase1=${P1H}h phase2=${P2H}h ==="

echo "--- PHASE 1: imitation / forward walk ---"
until ~/mjxenv/bin/python mjx/imitation.py --envs 4096 --hours "$P1H" --resume; do
    echo "PHASE1 RELAUNCH $(date)"; sleep 5
done

echo "--- PHASE 2: 2v2 self-play soccer ---"
until ~/mjxenv/bin/python mjx/train_soccer.py \
        --init mjx/checkpoints/walk_fwd.pkl --envs 1024 --hours "$P2H" --resume; do
    echo "PHASE2 RELAUNCH $(date)"; sleep 5
done

echo "=== PIPELINE done $(date) ==="
