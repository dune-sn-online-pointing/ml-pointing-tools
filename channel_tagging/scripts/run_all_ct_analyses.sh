#!/bin/bash
#
# Run comprehensive analysis on all successful CT model directories
#

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CT_BASE="/eos/user/e/evilla/dune/sn-tps/neural_networks/channel_tagging"
SCRIPT_DIR="$REPO_DIR/channel_tagging/ana"

echo "=========================================="
echo "Running CT Analysis on All Models"
echo "=========================================="
echo ""

# Find all directories with predictions
cd "$CT_BASE"

success_count=0
fail_count=0
skip_count=0

for dir in */; do
    dir_name="${dir%/}"
    
    # Skip archive directories
    if [[ "$dir_name" == _archive* ]]; then
        echo "⏭️  Skipping archive: $dir_name"
        ((skip_count++))
        continue
    fi
    
    # Check if predictions exist
    pred_file=$(find "$CT_BASE/$dir_name" -name "*predictions*.npz" 2>/dev/null | head -1)
    
    if [ -z "$pred_file" ]; then
        echo "⏭️  No predictions found in: $dir_name"
        ((skip_count++))
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "Processing: $dir_name"
    echo "=========================================="
    
    # Run analysis
    python3 "$SCRIPT_DIR/comprehensive_ct_analysis.py" "$CT_BASE/$dir_name"
    
    if [ $? -eq 0 ]; then
        echo "✅ Success: $dir_name"
        ((success_count++))
    else
        echo "❌ Failed: $dir_name"
        ((fail_count++))
    fi
done

echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "✅ Successful: $success_count"
echo "❌ Failed: $fail_count"
echo "⏭️  Skipped: $skip_count"
echo "=========================================="

