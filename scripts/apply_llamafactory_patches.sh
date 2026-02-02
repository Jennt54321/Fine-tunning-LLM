#!/bin/bash
# Apply custom patches to LLaMA-Factory submodule.
# Run from project root: ./scripts/apply_llamafactory_patches.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PATCHES_DIR="$PROJECT_ROOT/patches"
LLAMA_DIR="$PROJECT_ROOT/LLaMA-Factory"

if [ ! -d "$LLAMA_DIR" ]; then
    echo "Error: LLaMA-Factory not found. Run: git submodule update --init"
    exit 1
fi

cd "$LLAMA_DIR"
for patch in "$PATCHES_DIR"/*.patch; do
    if [ -f "$patch" ]; then
        echo "Applying $(basename "$patch")..."
        if git apply --check "$patch" 2>/dev/null; then
            git apply "$patch"
            echo "  Applied successfully."
        elif patch -p1 --forward < "$patch" 2>/dev/null; then
            echo "  Applied successfully (via patch)."
        else
            echo "  Skipped (may already be applied or incompatible with current LLaMA-Factory version)."
        fi
    fi
done
echo "Done."
