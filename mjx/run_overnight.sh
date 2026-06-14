#!/usr/bin/env bash
# Overnight training launcher. Kept as a foreground process by a long-lived
# wsl.exe on the Windows side, so the WSL2 distro stays up for the whole run.
# Auto-resumes from the last checkpoint if the trainer ever crashes.
#
#   wsl -d Ubuntu-24.04 -- bash mjx/run_overnight.sh [HOURS]
set -u
cd "$(dirname "$0")/.."                 # repo root
export MUJOCO_GL=egl
HOURS="${1:-10}"
echo "=== overnight run: ${HOURS}h, $(date) ==="
until ~/mjxenv/bin/python mjx/train_overnight.py \
        --envs 4096 --hours "$HOURS" --updates 100000000 \
        --save-every 100 --render-every 1000 \
        --ckpt mjx/checkpoints/overnight.pkl --resume; do
    echo "RELAUNCH-AFTER-CRASH $(date)"
    sleep 5
done
echo "=== finished, $(date) ==="
