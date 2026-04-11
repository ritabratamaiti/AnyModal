#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# VLM Training Launcher
# ═══════════════════════════════════════════════════════════════════════
# Run this script from a regular terminal (NOT inside the Copilot CLI snap)
# to have full GPU access.
#
# Usage:
#   Single GPU:  bash run_train.sh
#   Multi GPU:   bash run_train.sh --multi-gpu
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAMBA_ROOT="$HOME/snap/copilot-cli/17/micromamba"
ENV_NAME="vlm"

# Activate micromamba env
eval "$($MAMBA_ROOT/micromamba shell hook -s bash --root-prefix $MAMBA_ROOT)" 2>/dev/null
micromamba activate $ENV_NAME

echo "═══════════════════════════════════════════════════════"
echo "  VLM Training: Qwen2.5-1.5B + SigLIP"
echo "═══════════════════════════════════════════════════════"
echo ""

# Check GPU
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f'  GPU {i}: {name} ({mem:.0f}GB)')
else:
    print('  WARNING: No GPU detected! Training will be very slow.')
    print('  Make sure you are running this OUTSIDE the snap sandbox.')
"

echo ""

if [[ "${1:-}" == "--multi-gpu" ]]; then
    echo "Launching with accelerate (multi-GPU)..."
    accelerate launch \
        --num_processes 2 \
        --mixed_precision fp16 \
        "$SCRIPT_DIR/train_qwen_siglip_vlm.py"
else
    echo "Launching single-GPU training..."
    python "$SCRIPT_DIR/train_qwen_siglip_vlm.py"
fi
