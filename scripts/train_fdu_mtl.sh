#!/bin/bash
# Train MSP-DA on FDU-MTL Sentiment Analysis.
# Iterates over all 16 target domains and averages results.
# Usage: bash scripts/train_fdu_mtl.sh [target_domain]

DOMAINS=(MR apparel baby books camera_photo dvd electronics health_personal_care imdb kitchen_housewares magazines music software sports_outdoors toys_games video)

TARGET=${1:-all}

run() {
    DOMAIN=$1
    CONFIG="configs/fdu_mtl_${DOMAIN}.yaml"
    if [ ! -f "$CONFIG" ]; then
        echo "Config not found for ${DOMAIN}, skipping."
        return
    fi
    echo "=== Training MSP-DA: target=${DOMAIN} ==="
    python train.py --config ${CONFIG}
    echo "=== Evaluating: target=${DOMAIN} ==="
    python evaluate.py \
        --config ${CONFIG} \
        --checkpoint checkpoints/fdu_mtl_${DOMAIN}/best_model.pt
}

if [ "$TARGET" = "all" ]; then
    for dom in "${DOMAINS[@]}"; do
        run $dom
    done
else
    run ${TARGET}
fi
