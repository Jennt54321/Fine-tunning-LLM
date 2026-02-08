#!/usr/bin/env python3
"""
Evaluate GSM8K (main or socratic) from LLaMA-Factory generated_predictions.jsonl.

支援研究計劃的四項指標：
1. Reasoning (Accuracy Pass@1): 此腳本以 #### 答案比對計算。
2. Guidance Density: 每則回覆中 ? 的數量（Question count）。
3. Socratic Format Adherence: 是否符合 fine-tuned 輸出格式（Question? ** Answer with <<expr=result>>，每步一行，#### 結尾）。
4. Guidance Quality (Socratic Score): 使用 **Qwen2.5-14B** 評估時請用 `evaluate_with_qwen14b.py`。

Type	Models
Subject models	Qwen2.5-3B-Instruct (base + fine-tuned)
Evaluators	Qwen2.5-7B, 14B, 72B (and their MLX 4-bit variants)

若要以 LLM 從「拿掉 #### 的推理」算出答案再比對，請用 `evaluate_with_qwen14b.py`。

用法：
1. 先跑 predict（擇一）：
   llamafactory-cli train m3_eval.yaml
   llamafactory-cli train m3_socratic_eval_finetuned.yaml
   llamafactory-cli train m3_socratic_eval_promptonly.yaml

2. 再跑此腳本：
   python evaluate_gsm8k.py outputs/gsm8k_qwen_m3_eval
   python evaluate_gsm8k.py outputs/gsm8k_socratic_qwen_m3_eval_finetuned
   python evaluate_gsm8k.py outputs/gsm8k_socratic_qwen_m3_eval_promptonly
   python evaluate_gsm8k.py path/to/generated_predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the answer after the last '####' or '###' (GSM8K-style). Accepts both."""
    if not text:
        return None
    for sep in ("####", "###"):
        if sep in text:
            parts = re.split(r"\s*" + re.escape(sep) + r"\s*", text)
            if len(parts) >= 2:
                raw = parts[-1].strip().split("\n")[0].strip()
                if raw:
                    return raw.replace(",", "").strip()
    return None


def count_questions(text: str) -> int:
    """Count '?' in response (Guidance Density)."""
    return (text or "").count("?")


# Socratic format: Question? ** Answer with <<expr=result>>, one step per line, #### at end
SOCRATIC_PATTERN = re.compile(r"\?\s*\*\*\s+")  # ? ** 
GSM8K_COMPUTE_PATTERN = re.compile(r"<<[^>]+>>")  # <<expr=result>>


def check_socratic_format_adherence(text: str) -> tuple[bool, dict]:
    """
    Check if output follows the fine-tuned Socratic format:
    - Question? ** Answer (at least one step)
    - <<expr=result>> computation blocks
    - #### final answer
    - Steps on separate lines (one Q&A per line)

    Returns (pass: bool, details: dict).
    """
    if not text or not text.strip():
        return False, {"has_qa_steps": False, "has_compute": False, "has_final": False, "steps_per_line": False}

    has_qa = bool(SOCRATIC_PATTERN.search(text))
    has_compute = bool(GSM8K_COMPUTE_PATTERN.search(text))
    has_final = "####" in text or "###" in text

    # Steps on separate lines: each line should have at most one ? ** pattern, or we count lines with ? **
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    qa_lines = sum(1 for ln in lines if SOCRATIC_PATTERN.search(ln))
    steps_per_line = qa_lines >= 1 and (len(lines) >= qa_lines or qa_lines >= 1)

    # Pass if all key elements present
    pass_ = has_qa and has_compute and has_final

    details = {
        "has_qa_steps": has_qa,
        "has_compute": has_compute,
        "has_final": has_final,
        "steps_per_line": steps_per_line,
    }
    return pass_, details


def evaluate(predictions_path: str, verbose: bool = False, output_json: str | None = None) -> None:
    if os.path.isdir(predictions_path):
        pred_dir = predictions_path
        predictions_path = os.path.join(predictions_path, "generated_predictions.jsonl")
    else:
        pred_dir = os.path.dirname(predictions_path)
    if not os.path.isfile(predictions_path):
        raise FileNotFoundError(
            f"Predictions not found: {predictions_path}\n"
            "Run predict first, e.g. llamafactory-cli train m3_socratic_eval_finetuned.yaml"
        )

    correct = 0
    total = 0
    missing_pred = 0
    missing_label = 0
    question_counts: list[int] = []
    socratic_format_ok = 0
    per_sample: list[dict] = []

    with open(predictions_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = row.get("label") or ""
            pred = row.get("predict") or ""
            prompt = row.get("prompt", "")
            gt = extract_gsm8k_answer(label)
            ans = extract_gsm8k_answer(pred)
            if gt is None:
                missing_label += 1
                continue
            total += 1
            rule_correct = ans is not None and gt == ans
            if ans is None:
                missing_pred += 1
            else:
                if gt == ans:
                    correct += 1
            qc = count_questions(pred)
            socratic_pass, socratic_details = check_socratic_format_adherence(pred)
            question_counts.append(qc)
            if socratic_pass:
                socratic_format_ok += 1
            rec = {
                "index": idx,
                "gt": gt,
                "reasoning_correct_rule": rule_correct,
                "question_count": qc,
                "socratic_format_adherence": socratic_pass,
                "socratic_format_details": socratic_details,
                "socratic_score": None,
            }
            if output_json:
                rec["prompt"] = prompt
                rec["predict"] = pred
                rec["label"] = label
            per_sample.append(rec)

    n = len(question_counts)
    mean_q = sum(question_counts) / n if n else 0

    print(f"Predictions: {predictions_path}")
    print(f"Total (with valid label): {total}")
    if missing_label:
        print(f"Skipped (no ####/### in label): {missing_label}")
    if missing_pred:
        print(f"Missing prediction (no ####/### in model output): {missing_pred}")
    print()

    # 1. Reasoning — Accuracy (Pass@1)
    if total:
        acc = 100.0 * correct / total
        print(f"[Reasoning] Accuracy (Pass@1): {correct}/{total} = {acc:.2f}%")
    else:
        print("[Reasoning] Accuracy: N/A (no valid samples)")

    # 2. Guidance Density — Question count
    if n:
        print(f"[Guidance Density] Question count: mean = {mean_q:.2f}, min = {min(question_counts)}, max = {max(question_counts)}")
    else:
        print("[Guidance Density] Question count: N/A")

    # 3. Socratic Format Adherence (fine-tuned output pattern)
    if total:
        socratic_fmt_pct = 100.0 * socratic_format_ok / total
        print(f"[Socratic Format Adherence] Question? ** <<expr>>, ####: {socratic_format_ok}/{total} = {socratic_fmt_pct:.2f}%")
    else:
        print("[Socratic Format Adherence] N/A")

    # 4. Socratic Score — 需另用 LLM 評估
    print("[Guidance Quality] Socratic Score: N/A — use evaluate_with_qwen14b.py")

    if output_json:
        mean_q = sum(question_counts) / n if n else 0
        socratic_fmt_pct = 100.0 * socratic_format_ok / total if total else 0
        acc = 100.0 * correct / total if total else 0
        out = {
            "predictions_path": predictions_path,
            "n_samples": total,
            "summary": {
                "reasoning_accuracy_rule": acc,
                "question_count_mean": mean_q,
                "socratic_format_adherence_pct": socratic_fmt_pct,
                "socratic_score_mean": None,
            },
            "per_sample": per_sample,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\nPer-sample results written to {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GSM8K from generated_predictions.jsonl (accuracy, question count, socratic format adherence)."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="outputs/gsm8k_qwen_m3_eval",
        help="Path to eval output dir or to generated_predictions.jsonl",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-o", "--output-json", default=None, help="Write per-sample results to JSON (default: {path}/eval_rule_results.json)")
    args = parser.parse_args()
    out = args.output_json
    if out is None and os.path.isdir(args.path):
        out = os.path.join(args.path, "eval_rule_results.json")
    evaluate(args.path, verbose=args.verbose, output_json=out)


if __name__ == "__main__":
    main()
