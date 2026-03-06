#!/usr/bin/env python3
"""
Print duplicate answers (finetuned) and predictions that violate DAG format (prompt-only).
"""
from __future__ import annotations

import os
import sys

# Run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluate_bert import load_predictions
from evaluate_dag import _extract_response, compare_dags, parse_socratic_dag


def print_duplicate_answers(path: str, max_answer_chars: int = 500) -> None:
    """Load predictions from path, print each sample that has duplicate answers with the duplicate text."""
    if os.path.isdir(path):
        path = os.path.join(path, "generated_predictions.jsonl")
    rows = load_predictions(path)
    print(f"=== Duplicate answers (finetuned): {path} ({len(rows)} samples) ===\n")
    count = 0
    for i, row in enumerate(rows):
        pred = row.get("predict") or ""
        dag = parse_socratic_dag(pred)
        for step_ix, step in enumerate(dag.steps):
            if step.is_duplicate_answer:
                count += 1
                answer_text = _extract_response(step.text)
                if len(answer_text) > max_answer_chars:
                    answer_text = answer_text[:max_answer_chars] + "..."
                print(f"Sample index {i}, step {step_ix + 1} (duplicate answer):")
                print(f"  {answer_text}")
                print()
    if count == 0:
        print("No duplicate answers found.")
    else:
        print(f"Total duplicate-answer steps printed: {count}")


def print_format_violations(path: str, max_pred_chars: int = 2000) -> None:
    """Load predictions from path, print each sample that does not follow DAG format (format_score < 1 or missing ####)."""
    if os.path.isdir(path):
        path = os.path.join(path, "generated_predictions.jsonl")
    rows = load_predictions(path)
    print(f"=== Predictions not following DAG format (prompt-only): {path} ({len(rows)} samples) ===\n")
    count = 0
    for i, row in enumerate(rows):
        pred = row.get("predict") or ""
        dag = parse_socratic_dag(pred)
        metrics = compare_dags(dag)
        if metrics["format_score_mean"] < 1.0 or not metrics["terminal_present"]:
            count += 1
            print(f"Sample index {i}: format_score={metrics['format_score_mean']}, terminal_present={metrics['terminal_present']}")
            snippet = pred if len(pred) <= max_pred_chars else pred[:max_pred_chars] + "\n... [truncated]"
            print(snippet)
            print("\n" + "-" * 60 + "\n")
    if count == 0:
        print("All predictions follow the DAG format.")
    else:
        print(f"Total format-violating samples: {count}")


def main() -> None:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    finetuned = os.path.join(base, "outputs", "20260303", "gsm8k_socratic_qwen_eval_finetuned")
    promptonly = os.path.join(base, "outputs", "20260303", "gsm8k_socratic_qwen_eval_promptonly")

    print_duplicate_answers(finetuned)
    print("\n")
    print_format_violations(promptonly)


if __name__ == "__main__":
    main()
