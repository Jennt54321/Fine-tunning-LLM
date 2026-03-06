#!/usr/bin/env python3
"""
Structural DAG evaluation for GSM8K Socratic predictions.

Evaluates prediction format only (no comparison to reference label). Checks that
each prediction follows the requested Socratic format:

Format rules:
  1. Each step: question ends with "?", response part starts with "**" (same line: "Question? ** Response").
  2. If the response contains a calculation, it must be wrapped in <<expr=result>>; no calculation is fine.
  3. Final answer line must start with "####".
  4. Duplicate question: same question text (before "?") repeated across steps is penalized.
  5. Duplicate answer: same response text (after "**") repeated across steps is penalized.

Models each Socratic response as a linear DAG (chain of step nodes + terminal node).

Usage:
  python evaluate_dag.py outputs/gsm8k_socratic_qwen_m3_eval_finetuned
  python evaluate_dag.py outputs/gsm8k_socratic_qwen_m3_eval_promptonly
  python evaluate_dag.py path/to/predictions.jsonl --max-samples 50 -v
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from dataclasses import dataclass, field

from evaluate_bert import load_predictions


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StepNode:
    """One Q&A line in the Socratic format: Question? ** Response."""
    text: str
    has_question_mark: bool = False
    has_star_separator: bool = False
    has_expression: bool = False
    has_valid_expression: bool = False
    has_calculation: bool = False  # True if response contains calculation-like content (then <<expr=result>> required)
    is_duplicate_question: bool = False
    is_duplicate_answer: bool = False


@dataclass
class TerminalNode:
    """The #### answer line."""
    text: str
    has_hash_separator: bool = False
    has_numeric_answer: bool = False


@dataclass
class SocraticDAG:
    """Linear DAG: step[0] -> step[1] -> ... -> terminal."""
    steps: list[StepNode] = field(default_factory=list)
    terminal: TerminalNode | None = None


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_STAR_SEP = re.compile(r"\?\s*\*\*")              # ? **
_EXPRESSION = re.compile(r"<<[^>]+>>")             # <<...>>
_VALID_EXPRESSION = re.compile(r"<<[^=]+=[^>]+>>") # <<expr=result>>
_HASH_SEP = re.compile(r"#{3,4}\s*")               # ### or ####
_NUMERIC = re.compile(r"-?\d+\.?\d*")
# Heuristic: response contains digits and an operator (calculation-like)
_CALCULATION_LIKE = re.compile(r"\d+[\s]*[\+\-\*\/=][\s]*\d+|\d+[\s]*[\+\-\*\/=]|[\+\-\*\/=][\s]*\d+")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_response(line: str) -> str:
    """Extract the response part (text after '**') from a step line. Returns normalized string for comparison."""
    if "**" not in line:
        return ""
    parts = line.split("**", 1)
    return parts[1].strip() if len(parts) > 1 else ""


def _has_calculation_like_content(response: str) -> bool:
    """True if response contains something that looks like a calculation (digits + operator)."""
    if not response:
        return False
    # Remove <<...>> so we don't double-count; check the rest for raw arithmetic
    without_angle = re.sub(r"<<[^>]+>>", "", response)
    return bool(_CALCULATION_LIKE.search(without_angle))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_socratic_dag(text: str) -> SocraticDAG:
    """Parse a Socratic response into a SocraticDAG."""
    dag = SocraticDAG()
    if not text or not text.strip():
        return dag

    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()

    for line in lines:
        # Check if this is the terminal line (must start with ####)
        if _HASH_SEP.search(line):
            after_hash = _HASH_SEP.split(line, maxsplit=1)
            answer_part = after_hash[-1].strip() if len(after_hash) > 1 else ""
            dag.terminal = TerminalNode(
                text=line,
                has_hash_separator=True,
                has_numeric_answer=bool(_NUMERIC.search(answer_part)),
            )
            continue

        # Otherwise it's a step node: Question? ** Response
        has_q = "?" in line
        q_text = ""
        if has_q:
            q_text = line.split("?")[0].strip().lower()

        is_dup_q = bool(q_text and q_text in seen_questions)
        if q_text:
            seen_questions.add(q_text)

        response_text = _extract_response(line)
        response_key = response_text.lower().strip() if response_text else ""
        is_dup_a = bool(response_key and response_key in seen_answers)
        if response_key:
            seen_answers.add(response_key)

        has_calc = _has_calculation_like_content(response_text)

        node = StepNode(
            text=line,
            has_question_mark=has_q,
            has_star_separator=bool(_STAR_SEP.search(line)),
            has_expression=bool(_EXPRESSION.search(line)),
            has_valid_expression=bool(_VALID_EXPRESSION.search(line)),
            has_calculation=has_calc,
            is_duplicate_question=is_dup_q,
            is_duplicate_answer=is_dup_a,
        )
        dag.steps.append(node)

    return dag


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _step_format_score(step: StepNode) -> float:
    """Format score: question ?, response **, and if calculation present then <<expr=result>> required."""
    q = 1.0 if step.has_question_mark else 0.0
    star = 1.0 if step.has_star_separator else 0.0
    expr = 1.0 if step.has_expression else 0.0
    # Only require valid <<expr=result>> when response contains calculation-like content
    effective_valid_expr = 1.0 if (step.has_valid_expression or not step.has_calculation) else 0.0
    return (q + star + expr + effective_valid_expr) / 4.0


def compare_dags(pred_dag: SocraticDAG) -> dict:
    """Evaluate format compliance of pred_dag only. Returns format/duplicate/terminal metrics and a composite score."""
    pred_n = len(pred_dag.steps)

    # format_score_mean: average format score across prediction steps
    if pred_dag.steps:
        format_scores = [_step_format_score(s) for s in pred_dag.steps]
        format_score_mean = statistics.mean(format_scores)
    else:
        format_score_mean = 0.0

    # duplicate_question_ratio
    if pred_dag.steps:
        dup_q_count = sum(1 for s in pred_dag.steps if s.is_duplicate_question)
        duplicate_question_ratio = dup_q_count / len(pred_dag.steps)
    else:
        duplicate_question_ratio = 0.0

    # duplicate_answer_ratio
    if pred_dag.steps:
        dup_a_count = sum(1 for s in pred_dag.steps if s.is_duplicate_answer)
        duplicate_answer_ratio = dup_a_count / len(pred_dag.steps)
    else:
        duplicate_answer_ratio = 0.0

    # terminal_present (final answer must start with ####)
    terminal_present = pred_dag.terminal is not None and pred_dag.terminal.has_hash_separator

    # dag_similarity (weighted composite; prediction-only)
    dag_similarity = (
        0.80 * format_score_mean
        + 0.10 * (1.0 - duplicate_question_ratio)
        + 0.10 * (1.0 - duplicate_answer_ratio)
    )

    return {
        "pred_step_count": pred_n,
        "format_score_mean": round(format_score_mean, 4),
        "duplicate_question_ratio": round(duplicate_question_ratio, 4),
        "duplicate_answer_ratio": round(duplicate_answer_ratio, 4),
        "terminal_present": terminal_present,
        "dag_similarity": round(dag_similarity, 4),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate(
    path: str,
    max_samples: int | None = None,
    output_json: str | None = None,
    verbose: bool = False,
) -> None:
    """Load predictions, parse DAGs, compare, summarize, and save."""
    pred_path = path
    if os.path.isdir(path):
        pred_path = os.path.join(path, "generated_predictions.jsonl")

    rows = load_predictions(path)
    if max_samples is not None:
        rows = rows[:max_samples]

    n = len(rows)
    print(f"Predictions: {pred_path}")
    print(f"Samples: {n}")
    print()

    per_sample = []
    for i, row in enumerate(rows):
        pred_text = row.get("predict") or ""

        pred_dag = parse_socratic_dag(pred_text)
        metrics = compare_dags(pred_dag)

        rec = {"index": i, **metrics}
        if verbose:
            rec["predict"] = pred_text
            rec["label"] = row.get("label") or ""
        per_sample.append(rec)

    # Aggregate summary
    summary = {}
    if per_sample:
        for key in [
            "format_score_mean",
            "duplicate_question_ratio",
            "duplicate_answer_ratio",
            "dag_similarity",
        ]:
            vals = [r[key] for r in per_sample]
            summary[f"{key}_mean"] = round(statistics.mean(vals), 4)
            if len(vals) > 1:
                summary[f"{key}_std"] = round(statistics.stdev(vals), 4)
            else:
                summary[f"{key}_std"] = 0.0

        terminal_count = sum(1 for r in per_sample if r["terminal_present"])
        summary["terminal_present_pct"] = round(100.0 * terminal_count / len(per_sample), 2)

    # Print summary
    print("[DAG Structural Evaluation]")
    if summary:
        print(f"  Format score:            {summary['format_score_mean_mean']:.4f} ± {summary['format_score_mean_std']:.4f}")
        print(f"  Duplicate question ratio: {summary['duplicate_question_ratio_mean']:.4f} ± {summary['duplicate_question_ratio_std']:.4f}")
        print(f"  Duplicate answer ratio:  {summary['duplicate_answer_ratio_mean']:.4f} ± {summary['duplicate_answer_ratio_std']:.4f}")
        print(f"  DAG similarity:          {summary['dag_similarity_mean']:.4f} ± {summary['dag_similarity_std']:.4f}")
        print(f"  Terminal present:        {summary['terminal_present_pct']:.1f}%")
    else:
        print("  No valid samples")

    if verbose:
        print("\n[Per-sample details]")
        for rec in per_sample:
            print(
                f"  #{rec['index']}: steps={rec['pred_step_count']} "
                f"fmt={rec['format_score_mean']:.4f} "
                f"dag={rec['dag_similarity']:.4f} dup_q={rec['duplicate_question_ratio']:.4f} dup_a={rec['duplicate_answer_ratio']:.4f} "
                f"terminal={'Y' if rec['terminal_present'] else 'N'}"
            )

    # Save JSON
    if output_json:
        result = {
            "predictions_path": pred_path,
            "n_samples": len(per_sample),
            "summary": summary,
            "per_sample": per_sample,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nResults written to {output_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Structural DAG evaluation for GSM8K Socratic predictions."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="outputs/gsm8k_socratic_qwen_m3_eval_finetuned",
        help="Path to eval output dir or generated_predictions.jsonl",
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
        help="Output path (default: {path}/eval_dag_results.json)",
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
            output_json = os.path.join(args.path, "eval_dag_results.json")
        elif os.path.isfile(args.path):
            output_json = os.path.join(os.path.dirname(args.path), "eval_dag_results.json")
        else:
            output_json = os.path.join(args.path, "eval_dag_results.json")

    evaluate(
        path=args.path,
        max_samples=args.max_samples,
        output_json=output_json,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
