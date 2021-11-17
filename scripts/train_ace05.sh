#!/bin/bash
# Train MSP-DA on ACE-05 Event Detection for all three target domains.
# Usage: bash scripts/train_ace05.sh [bc|cts|wl|all]

TARGET=${1:-all}

run() {
    DOMAIN=$1
    CONFIG="configs/ace05_${DOMAIN}.yaml"
    echo "=== Training MSP-DA: source=bn+nw  target=${DOMAIN} ==="
    python train.py --config ${CONFIG}
    echo "=== Evaluating: target=${DOMAIN} ==="
    python evaluate.py \
        --config ${CONFIG} \
        --checkpoint checkpoints/ace05_${DOMAIN}/best_model.pt
}

if [ "$TARGET" = "all" ]; then
    run bc
    run cts
    run wl
else
    run ${TARGET}
fi
