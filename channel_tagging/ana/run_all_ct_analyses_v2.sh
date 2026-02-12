#!/bin/bash
# Run comprehensive CT analysis V2 on all successful models

BASE_DIR="/eos/user/e/evilla/dune/sn-tps/neural_networks/channel_tagging"
SCRIPT="/afs/cern.ch/work/e/evilla/private/dune/ml-pointing-tools/channel_tagging/ana/comprehensive_ct_analysis_v2.py"

# List of successful CT model directories
MODELS=(
    "ct_volume_v52_batch_reload_20251116_101125"
    "ct_volume_v72_deeper_10k_20251124_005847"
    "ct_volume_v78_dario_10k_20251123_153908"
    "three_plane_v70_20251119_174031"
    "v77_dario_batch_5k_20251121_021945"
    "v77_dario_batch_5k_20251121_073847"
)

echo "=========================================="
echo "Running Comprehensive CT Analysis V2"
echo "=========================================="
echo ""

for model in "${MODELS[@]}"; do
    model_dir="$BASE_DIR/$model"
    
    if [ ! -d "$model_dir" ]; then
        echo "❌ Directory not found: $model_dir"
        continue
    fi
    
    echo "Processing: $model"
    echo "---"
    
    python3 "$SCRIPT" "$model_dir"
    
    if [ $? -eq 0 ]; then
        echo "✅ Success: $model"
    else
        echo "❌ Failed: $model"
    fi
    
    echo ""
done

echo "=========================================="
echo "All analyses complete!"
echo "=========================================="
