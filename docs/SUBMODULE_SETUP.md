# LLaMA-Factory Submodule Setup

This project uses [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) as a git submodule with custom patches for Apple Silicon (MPS) compatibility.

## Initial Setup (New Clone)

```bash
# 1. Clone the repo (if you haven't already)
git clone --recurse-submodules https://github.com/YOUR_USERNAME/Fine-tuning-LLM.git
cd Fine-tuning-LLM

# 2. If you cloned without --recurse-submodules, init and update:
# git submodule update --init

# 3. Apply custom patches (MPS fix for metric.py)
./scripts/apply_llamafactory_patches.sh

# 4. Install LLaMA-Factory (from project root)
pip install -e LLaMA-Factory
```

## After Pulling Updates

If the submodule reference was updated:

```bash
git submodule update --init
./scripts/apply_llamafactory_patches.sh
```

## Custom Patches

| Patch | Purpose |
|-------|---------|
| `patches/llamafactory-metric-mps-fix.patch` | Fixes `OverflowError` in `tokenizer.decode` on MPS (Apple Silicon). Handles `pad_token_id is None` and clips token IDs to valid range. |

## Updating LLaMA-Factory

```bash
cd LLaMA-Factory
git fetch origin
git checkout main
git pull origin main
cd ..

# Re-apply patches (may need manual resolution if upstream changed)
./scripts/apply_llamafactory_patches.sh

# Commit the submodule update
git add LLaMA-Factory
git commit -m "Update LLaMA-Factory submodule"
```
