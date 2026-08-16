#!/bin/bash
set -e

echo "checking nvidia gpu..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "detected ${VRAM}MB VRAM"
    if [ "$VRAM" -ge 16000 ]; then
        echo "tier: large (16GB+) - can run larger models"
    elif [ "$VRAM" -ge 8000 ]; then
        echo "tier: medium (8GB) - Qwen2.5-3B Q4_K_M + whisper small"
    else
        echo "tier: small (<8GB) - Qwen2.5-1.5B Q4_K_M + whisper tiny"
    fi
else
    echo "no nvidia gpu detected, will use CPU inference"
fi
