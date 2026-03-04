#!/usr/bin/env python3
"""
Run LLaMA-Factory vllm_infer.py for Socratic eval (prompt-only and/or fine-tuned).
Produces the same generated_predictions.jsonl as Step 3 without using the trainer,
so it avoids the batch_decode TypeError when saving predictions.

Defaults: Colab paths (repo /content/Fine-tunning-LLM, outputs /content/drive/MyDrive/llm_outputs).
For local: set SOCRATIC_REPO_ROOT=auto and SOCRATIC_OUTPUT_DIR=/path/to/outputs (or leave OUTPUT_DIR to use Drive path if you mount it).

Usage (from repo root, e.g. /content/Fine-tunning-LLM in Colab):
  python scripts/run_vllm_predict.py [promptonly|finetuned|both] [--max_samples N]
  # Quick test with 3 samples:
  python scripts/run_vllm_predict.py both --max_samples 3

Requires: pip install vllm, and run from an environment where LLaMA-Factory is
installed (e.g. pip install -e ./LLaMA-Factory from repo root).
"""
import argparse
import os
import subprocess
import sys

# Colab defaults (override with env for local)
REPO_ROOT = os.environ.get("SOCRATIC_REPO_ROOT", "/content/Fine-tunning-LLM")
if REPO_ROOT == "auto" or (REPO_ROOT == "/content/Fine-tunning-LLM" and not os.path.isdir(REPO_ROOT)):
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("SOCRATIC_DATA_DIR", os.path.join(REPO_ROOT, "data", "custom"))
OUTPUT_BASE = os.environ.get("SOCRATIC_OUTPUT_DIR", "/content/drive/MyDrive/llm_outputs")
ADAPTER_PATH = os.environ.get(
    "SOCRATIC_ADAPTER_PATH",
    os.path.join(OUTPUT_BASE, "socratic_qwen25-3b-instruct"),
)

LLAMA_FACTORY_DIR = os.path.join(REPO_ROOT, "LLaMA-Factory")
PROMPTONLY_SAVE = os.path.join(OUTPUT_BASE, "gsm8k_socratic_qwen_eval_promptonly", "generated_predictions.jsonl")
FINETUNED_SAVE = os.path.join(OUTPUT_BASE, "gsm8k_socratic_qwen_eval_finetuned", "generated_predictions.jsonl")

MODEL = "Qwen/Qwen2.5-3B-Instruct"
DATASET = "gsm8k_socratic_test"
TEMPLATE = "qwen"
CUTOFF_LEN = 2048
TEMPERATURE = 0  # greedy (match do_sample: false in yaml)


def _run(mode: str, max_samples: int | None = None) -> int:
    os.makedirs(os.path.dirname(PROMPTONLY_SAVE), exist_ok=True)
    os.makedirs(os.path.dirname(FINETUNED_SAVE), exist_ok=True)

    cmd_base = [
        sys.executable,
        os.path.join(LLAMA_FACTORY_DIR, "scripts", "vllm_infer.py"),
        "--model_name_or_path", MODEL,
        "--dataset", DATASET,
        "--dataset_dir", DATA_DIR,
        "--template", TEMPLATE,
        "--cutoff_len", str(CUTOFF_LEN),
        "--temperature", str(TEMPERATURE),
    ]
    if max_samples is not None:
        cmd_base += ["--max_samples", str(max_samples)]

    if mode in ("promptonly", "both"):
        cmd = cmd_base + ["--save_name", PROMPTONLY_SAVE]
        print("Running prompt-only prediction ...")
        print(" ", " ".join(cmd))
        ret = subprocess.run(cmd, cwd=LLAMA_FACTORY_DIR)
        if ret.returncode != 0:
            return ret.returncode

    if mode in ("finetuned", "both"):
        cmd = cmd_base + [
            "--adapter_name_or_path", ADAPTER_PATH,
            "--save_name", FINETUNED_SAVE,
        ]
        print("Running fine-tuned prediction ...")
        print(" ", " ".join(cmd))
        ret = subprocess.run(cmd, cwd=LLAMA_FACTORY_DIR)
        if ret.returncode != 0:
            return ret.returncode

    return 0


def main():
    parser = argparse.ArgumentParser(description="Run vllm_infer for Socratic eval (prompt-only and/or fine-tuned).")
    parser.add_argument(
        "mode",
        nargs="?",
        default="both",
        choices=("promptonly", "finetuned", "both"),
        help="Which prediction to run",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        metavar="N",
        help="Limit to N samples (e.g. 3 for a quick test)",
    )
    args = parser.parse_args()
    if not os.path.isdir(LLAMA_FACTORY_DIR):
        print("LLaMA-Factory not found at", LLAMA_FACTORY_DIR, file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(DATA_DIR):
        print("Data dir not found:", DATA_DIR, file=sys.stderr)
        sys.exit(1)
    sys.exit(_run(args.mode, args.max_samples))


if __name__ == "__main__":
    main()
