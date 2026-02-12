
#!/bin/bash
# Channel Tagging Training Script
# This script runs the channel tagging classifier

set -e

# Source initialization
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_DIR/scripts/init.sh"

# Default configuration
INPUT_JSON="$JSON_DIR/channel_tagging/production_training.json"
OUTPUT_FOLDER="$OUTPUT_DIR/channel_tagging"

print_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Run channel tagging classifier training.

Options:
    -j, --input_json <file>     Path to JSON configuration file
    -o, --output_folder <dir>   Output directory for results
    -h, --help                  Show this help message

Examples:
    $0 -j json/channel_tagging/production_training.json
    $0 -j config.json -o /path/to/output

EOF
    exit 0
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--input_json)
            INPUT_JSON="$2"
            shift 2
            ;;
        -o|--output_folder)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            ;;
        *)
            echo "Error: Unknown option $1"
            print_help
            ;;
    esac
done

# Validate inputs
check_file "$INPUT_JSON"
ensure_dir "$OUTPUT_FOLDER"

echo "[INFO] Starting Channel Tagging Training"
echo "[INFO] Configuration: $INPUT_JSON"
echo "[INFO] Output: $OUTPUT_FOLDER"
echo "[INFO] Running training script..."

# Run training
python3 "$REPO_DIR/channel_tagging/models/ct_training.py" \
    --input_json "$INPUT_JSON" \
    --output_folder "$OUTPUT_FOLDER"
