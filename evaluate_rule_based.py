#!/usr/bin/env python3
"""
Evaluate GSM8K (main or socratic) from LLaMA-Factory generated_predictions.jsonl.

支援研究計劃的四項指標：
1. Reasoning (Accuracy Pass@1): 此腳本以 format-free 答案提取比對（#### 優先，其次 ** 數字 **、最後 <<expr=result>>、最後一行的數字）。
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
   llamafactory-cli train socratic_finetuned_eval.yaml
   llamafactory-cli train socratic_promptonly_eval.yaml

2. 再跑此腳本：
   python evaluate_gsm8k.py outputs/gsm8k_qwen_m3_eval
   python evaluate_gsm8k.py outputs/gsm8k_socratic_qwen_eval_finetuned
   python evaluate_gsm8k.py outputs/gsm8k_socratic_qwen_eval_promptonly
   python evaluate_gsm8k.py path/to/generated_predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison: remove commas, strip, handle 18 vs 18.0."""
    if not s:
        return ""
    s = s.strip().replace(",", "").replace(" ", "")
    # Remove trailing .0 for integer equivalence (18.0 == 18)
    if "." in s:
        try:
            f = float(s)
            if f == int(f):
                return str(int(f))
            return s
        except ValueError:
            pass
    return s


def extract_gsm8k_answer(text: str) -> tuple[str | None, str]:
    """
    Extract the final numerical answer from model output, format-agnostic.

    Tries multiple strategies (in order):
    1. After #### or ### (GSM8K standard)
    2. ** number ** or ** number at end (e.g. "** 24 miles **")
    3. Last <<expr=result>> result (e.g. <<9*2=18>> -> 18)
    4. Last number in the last 2 lines (common when format is wrong)

    Returns (extracted_answer, method) where method is for debugging.
    """
    if not text or not text.strip():
        return None, "empty"

    # 1. #### or ### (primary)
    for sep in ("####", "###"):
        if sep in text:
            parts = re.split(r"\s*" + re.escape(sep) + r"\s*", text)
            if len(parts) >= 2:
                raw = parts[-1].strip().split("\n")[0].strip()
                if raw:
                    ans = _extract_first_number(raw)
                    if ans is not None:
                        return ans, f"sep_{sep}"

    # 2. ** number ** or ** number at end (original model often uses this)
    star_match = re.search(r"\*\*\s*([-\d.,]+(?:\s*[%\$\w]+)?)\s*\*?\*?\s*$", text)
    if star_match:
        ans = _extract_first_number(star_match.group(1))
        if ans is not None:
            return ans, "star_bold"

    # 3. Last <<expr=result>> — take the result part
    compute_matches = list(re.finditer(r"<<[^=]+=([^>]+)>>", text))
    if compute_matches:
        last_result = compute_matches[-1].group(1).strip()
        ans = _extract_first_number(last_result)
        if ans is not None:
            return ans, "last_compute"

    # 4. Last number in last 2 lines (format-free fallback)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for line in reversed(lines[-2:] if len(lines) >= 2 else lines):
        numbers = re.findall(r"-?\d+\.?\d*", line)
        if numbers:
            # Take last number from the line (often the final answer)
            last_num = numbers[-1]
            # Sanity: avoid very large numbers that are likely intermediates
            try:
                n = float(last_num)
                if abs(n) < 1e12:
                    return last_num, "last_number"
            except ValueError:
                pass

    return None, "not_found"


def _extract_first_number(s: str) -> str | None:
    """Extract the first number (int or decimal) from a string. Handles $18, 18%, 18.5, etc."""
    s = s.strip().replace(",", "")
    match = re.search(r"-?\d+\.?\d*", s)
    if match:
        return match.group(0)
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
            "Run predict first, e.g. llamafactory-cli train socratic_finetuned_eval.yaml"
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
            gt, _ = extract_gsm8k_answer(label)
            ans, ans_method = extract_gsm8k_answer(pred)
            if gt is None:
                missing_label += 1
                continue
            total += 1
            # Normalize for comparison: 18 vs 18.0, 1,000 vs 1000
            gt_norm = normalize_answer(gt)
            ans_norm = normalize_answer(ans) if ans else ""
            rule_correct = ans is not None and gt_norm == ans_norm
            if ans is None:
                missing_pred += 1
            else:
                if gt_norm == ans_norm:
                    correct += 1
            qc = count_questions(pred)
            socratic_pass, socratic_details = check_socratic_format_adherence(pred)
            question_counts.append(qc)
            if socratic_pass:
                socratic_format_ok += 1
            rec = {
                "index": idx,
                "gt": gt,
                "pred_extracted": ans,
                "extraction_method": ans_method,
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
        print(f"Skipped (no extractable answer in label): {missing_label}")
    if missing_pred:
        print(f"Could not extract answer from model output: {missing_pred}")
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
