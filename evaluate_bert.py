#!/usr/bin/env python3
"""
BERTScore evaluation for hallucination/coverage detection on GSM8K Socratic predictions.

Computes token-level semantic similarity (precision/recall/F1) between prediction
and reference using BERT embeddings. Tolerates different-but-valid reasoning paths
while catching genuine hallucinations.

Usage:
  python evaluate_bert.py outputs/gsm8k_socratic_qwen_m3_eval_finetuned
  python evaluate_bert.py outputs/... --step-level
  python evaluate_bert.py outputs/... --model roberta-large --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys


def strip_final_answer(text: str) -> str:
    """Remove #### / ### final answer line, keeping only the reasoning."""
    if not text:
        return ""
    for sep in ("####", "###"):
        if sep in text:
            parts = re.split(r"\s*" + re.escape(sep) + r"\s*", text)
            return parts[0].strip()
    return text


def strip_socratic_markers(text: str) -> str:
    """Replace <<expr=result>> with just the result value; remove ** markers."""
    # <<16-3-4=9>> -> 9
    text = re.sub(r"<<[^=]+=([^>]+)>>", r"\1", text)
    # Remove ** markers
    text = text.replace("**", "")
    return text


# Max chars per text to avoid tokenizer OverflowError (int too big to convert).
# DeBERTa/transformers fast tokenizer can overflow on large model_max_length; keep conservative.
BERTSCORE_MAX_CHARS = 1024


def truncate_for_bertscore(text: str, max_chars: int = BERTSCORE_MAX_CHARS) -> str:
    """Truncate to max_chars to avoid tokenizer overflow on very long inputs."""
    if not text or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def preprocess_text(text: str, strip_markers: bool = True) -> str:
    """Pipeline: strip final answer -> optionally strip markers -> strip whitespace."""
    text = strip_final_answer(text)
    if strip_markers:
        text = strip_socratic_markers(text)
    return text.strip()


def split_into_steps(text: str) -> list[str]:
    """Split by newline, filter empties."""
    return [line.strip() for line in text.split("\n") if line.strip()]


def resolve_device(requested: str) -> str:
    """Device detection: auto resolves to cuda > cpu. MPS is opt-in only."""
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "mps":
        if not torch.backends.mps.is_available():
            print("Warning: MPS not available, falling back to CPU", file=sys.stderr)
            return "cpu"
        return "mps"
    return requested


def load_predictions(path: str) -> list[dict]:
    """Load JSONL predictions. Accepts dir or file path."""
    if os.path.isdir(path):
        path = os.path.join(path, "generated_predictions.jsonl")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Predictions not found: {path}\n"
            "Run predict first, e.g. llamafactory-cli train socratic_finetuned_eval.yaml"
        )
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_bertscore_batch(
    preds: list[str],
    refs: list[str],
    model_type: str,
    device: str,
    batch_size: int,
) -> tuple[list[float], list[float], list[float]]:
    """Wrapper around bert_score.score(). Returns (precisions, recalls, f1s)."""
    import importlib
    _score_mod = importlib.import_module("bert_score.score")

    # Cap tokenizer.model_max_length to avoid OverflowError in set_truncation_and_padding (Rust/32-bit).
    # Must patch score module's get_tokenizer (score() uses that reference, not utils).
    _original_get_tokenizer = _score_mod.get_tokenizer
    def _patched_get_tokenizer(model_type_arg, use_fast_tokenizer=True):
        tok = _original_get_tokenizer(model_type_arg, use_fast_tokenizer)
        max_len = getattr(tok, "model_max_length", None)
        if max_len is None or (isinstance(max_len, int) and max_len > 512):
            tok.model_max_length = 512
        return tok
    _score_mod.get_tokenizer = _patched_get_tokenizer
    try:
        from bert_score import score
        P, R, F1 = score(
            preds,
            refs,
            model_type=model_type,
            device=device,
            batch_size=batch_size,
            verbose=False,
        )
        return P.tolist(), R.tolist(), F1.tolist()
    finally:
        _score_mod.get_tokenizer = _original_get_tokenizer


def _pad_steps(pred_steps: list[str], label_steps: list[str]) -> tuple[list[str], list[str]]:
    """Pad shorter list by repeating last step. Returns (padded_pred, padded_label)."""
    max_len = max(len(pred_steps), len(label_steps))
    pred_padded = list(pred_steps)
    label_padded = list(label_steps)
    while len(pred_padded) < max_len:
        pred_padded.append(pred_padded[-1])
    while len(label_padded) < max_len:
        label_padded.append(label_padded[-1])
    return pred_padded, label_padded


def compute_step_level_bertscore_batched(
    preds: list[str],
    refs: list[str],
    model_type: str,
    device: str,
    batch_size: int,
) -> list[dict]:
    """Batch-compute step-level BERTScore across all samples. Returns list of per-sample dicts."""
    all_pred_steps: list[str] = []
    all_ref_steps: list[str] = []
    sample_info: list[tuple[int, int, int, int]] = []  # (start, end, n_pred, n_label)

    for pred_text, ref_text in zip(preds, refs):
        pred_steps = split_into_steps(pred_text)
        label_steps = split_into_steps(ref_text)
        n_pred = len(pred_steps)
        n_label = len(label_steps)

        if not pred_steps or not label_steps:
            sample_info.append((-1, -1, n_pred, n_label))
            continue

        pred_padded, label_padded = _pad_steps(pred_steps, label_steps)
        start = len(all_pred_steps)
        all_pred_steps.extend(pred_padded)
        all_ref_steps.extend(label_padded)
        sample_info.append((start, len(all_pred_steps), n_pred, n_label))

    if not all_pred_steps:
        return [
            {
                "n_pred_steps": info[2],
                "n_label_steps": info[3],
                "step_scores": [],
                "step_f1_mean": 0.0,
            }
            for info in sample_info
        ]

    P, R, F1 = compute_bertscore_batch(
        all_pred_steps, all_ref_steps, model_type, device, batch_size
    )

    step_results = []
    for info in sample_info:
        start, end, n_pred, n_label = info
        if start < 0:
            step_results.append({
                "n_pred_steps": n_pred,
                "n_label_steps": n_label,
                "step_scores": [],
                "step_f1_mean": 0.0,
            })
            continue
        step_scores = [
            {
                "step": j,
                "precision": round(P[start + j], 4),
                "recall": round(R[start + j], 4),
                "f1": round(F1[start + j], 4),
            }
            for j in range(end - start)
        ]
        f1_slice = F1[start:end]
        step_results.append({
            "n_pred_steps": n_pred,
            "n_label_steps": n_label,
            "step_scores": step_scores,
            "step_f1_mean": round(statistics.mean(f1_slice), 4),
        })

    return step_results


def evaluate(
    path: str,
    model_type: str = "microsoft/deberta-xlarge-mnli",
    device: str = "auto",
    batch_size: int = 32,
    step_level: bool = False,
    strip_markers: bool = True,
    max_samples: int | None = None,
    output_json: str | None = None,
    verbose: bool = False,
) -> None:
    """Main orchestration: load -> preprocess -> compute -> summarize -> save."""
    # Resolve paths
    pred_path = path
    if os.path.isdir(path):
        pred_dir = path
        pred_path = os.path.join(path, "generated_predictions.jsonl")
    else:
        pred_dir = os.path.dirname(path)

    rows = load_predictions(path)
    if max_samples is not None:
        rows = rows[:max_samples]

    device = resolve_device(device)
    n = len(rows)
    print(f"Predictions: {pred_path}")
    print(f"BERTScore model: {model_type}")
    print(f"Device: {device}")
    print(f"Samples: {n}")
    print()

    # Preprocess and track indices with originally empty pred/ref (for score zeroing)
    preds = []
    refs = []
    valid_rows = []
    empty_score_indices: set[int] = set()
    for row in rows:
        pred = row.get("predict") or ""
        label = row.get("label") or ""
        pred_proc = truncate_for_bertscore(preprocess_text(pred, strip_markers=strip_markers))
        label_proc = truncate_for_bertscore(preprocess_text(label, strip_markers=strip_markers))
        if not pred_proc or not label_proc:
            empty_score_indices.add(len(preds))
        preds.append(pred_proc if pred_proc else " ")
        refs.append(label_proc if label_proc else " ")
        valid_rows.append(row)

    # Compute sample-level BERTScore
    print("Computing BERTScore...")
    P, R, F1 = compute_bertscore_batch(preds, refs, model_type, device, batch_size)

    # Set scores to 0.0 for originally empty texts (use precomputed indices)
    for i in empty_score_indices:
        P[i] = 0.0
        R[i] = 0.0
        F1[i] = 0.0

    # Step-level (optional) - single batched call instead of N separate calls
    step_results = []
    if step_level:
        print("Computing step-level BERTScore...")
        step_results = compute_step_level_bertscore_batched(
            preds, refs, model_type, device, batch_size
        )

    # Build per-sample records
    per_sample = []
    for i, row in enumerate(valid_rows):
        rec = {
            "index": i,
            "bertscore_precision": round(P[i], 4),
            "bertscore_recall": round(R[i], 4),
            "bertscore_f1": round(F1[i], 4),
        }
        if step_level and step_results:
            rec["step_level"] = step_results[i]
        rec["predict"] = row.get("predict", "")
        rec["label"] = row.get("label", "")
        per_sample.append(rec)

    # Summary stats
    p_vals = [r["bertscore_precision"] for r in per_sample]
    r_vals = [r["bertscore_recall"] for r in per_sample]
    f1_vals = [r["bertscore_f1"] for r in per_sample]

    summary = {}
    if p_vals:
        summary["bertscore_precision_mean"] = round(statistics.mean(p_vals), 4)
        summary["bertscore_precision_std"] = round(statistics.stdev(p_vals), 4) if len(p_vals) > 1 else 0.0
        summary["bertscore_recall_mean"] = round(statistics.mean(r_vals), 4)
        summary["bertscore_recall_std"] = round(statistics.stdev(r_vals), 4) if len(r_vals) > 1 else 0.0
        summary["bertscore_f1_mean"] = round(statistics.mean(f1_vals), 4)
        summary["bertscore_f1_std"] = round(statistics.stdev(f1_vals), 4) if len(f1_vals) > 1 else 0.0
        summary["bertscore_f1_median"] = round(statistics.median(f1_vals), 4)

    if step_level and step_results:
        step_f1_means = [s["step_f1_mean"] for s in step_results if s["step_scores"]]
        if step_f1_means:
            summary["step_bertscore_f1_mean"] = round(statistics.mean(step_f1_means), 4)

    # Print summary
    print("[BERTScore Evaluation]")
    if summary:
        print(f"  Precision: {summary['bertscore_precision_mean']:.4f} ± {summary['bertscore_precision_std']:.4f}")
        print(f"  Recall:    {summary['bertscore_recall_mean']:.4f} ± {summary['bertscore_recall_std']:.4f}")
        print(f"  F1:        {summary['bertscore_f1_mean']:.4f} ± {summary['bertscore_f1_std']:.4f} (median: {summary['bertscore_f1_median']:.4f})")
        if "step_bertscore_f1_mean" in summary:
            print(f"  Step-level F1 mean: {summary['step_bertscore_f1_mean']:.4f}")
    else:
        print("  No valid samples")

    if verbose:
        print("\n[Per-sample details]")
        for rec in per_sample:
            print(f"  #{rec['index']}: P={rec['bertscore_precision']:.4f} R={rec['bertscore_recall']:.4f} F1={rec['bertscore_f1']:.4f}")
            if "step_level" in rec:
                sl = rec["step_level"]
                print(f"    Steps: pred={sl['n_pred_steps']} label={sl['n_label_steps']} step_f1_mean={sl['step_f1_mean']:.4f}")

    # Save JSON
    if output_json:
        result = {
            "predictions_path": pred_path,
            "model_type": model_type,
            "n_samples": len(per_sample),
            "summary": summary,
            "per_sample": per_sample,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nResults written to {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="BERTScore evaluation for hallucination/coverage detection on GSM8K Socratic predictions."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="outputs/gsm8k_socratic_qwen_m3_eval_finetuned",
        help="Path to eval output dir or generated_predictions.jsonl",
    )
    parser.add_argument(
        "--model",
        default="microsoft/deberta-xlarge-mnli",
        help="BERTScore model (default: microsoft/deberta-xlarge-mnli)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device: auto resolves to cuda > cpu; MPS is opt-in (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for BERTScore (default: 32)",
    )
    parser.add_argument(
        "--step-level",
        action="store_true",
        help="Also compute per-step BERTScore (splits by newline)",
    )
    parser.add_argument(
        "--no-strip-markers",
        action="store_true",
        help="Keep <<expr=result>> and ** markers",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap samples for quick testing",
    )
    parser.add_argument(
        "-o", "--output-json",
        default=None,
        help="Output path (default: {path}/eval_bert_results.json)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print per-sample details",
    )
    args = parser.parse_args()

    output_json = args.output_json
    if output_json is None:
        if os.path.isdir(args.path):
            output_json = os.path.join(args.path, "eval_bert_results.json")
        elif os.path.isfile(args.path):
            output_json = os.path.join(os.path.dirname(args.path), "eval_bert_results.json")
        else:
            output_json = os.path.join(args.path, "eval_bert_results.json")

    evaluate(
        path=args.path,
        model_type=args.model,
        device=args.device,
        batch_size=args.batch_size,
        step_level=args.step_level,
        strip_markers=not args.no_strip_markers,
        max_samples=args.max_samples,
        output_json=output_json,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
