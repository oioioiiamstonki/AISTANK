#!/usr/bin/env bash
# Phase 2 only: self-play soccer, warm-started from the Phase-1 walker.
# Auto-resumes from mjx/checkpoints/soccer.pkl on crash.
#   wsl -d Ubuntu-24.04 -- bash mjx/run_soccer.sh [ENVS] [HOURS]
set -u
cd "$(dirname "$0")/.."
export MUJOCO_GL=egl
ENVS="${1:-256}"      # 256 fits 16GB VRAM (4 players/env, 99-DoF system)
HRS="${2:-10}"
echo "=== SOCCER start $(date): envs=${ENVS} hours=${HRS} ==="
until ~/mjxenv/bin/python mjx/train_soccer.py \
        --init mjx/checkpoints/walk_fwd.pkl --envs "$ENVS" --hours "$HRS" --resume; do
    echo "RELAUNCH $(date)"; sleep 5
done
echo "=== SOCCER done $(date) ==="
