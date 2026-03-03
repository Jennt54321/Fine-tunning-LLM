#!/usr/bin/env python3
"""
Structural DAG evaluation for GSM8K Socratic predictions.

Models each Socratic response as a linear DAG (chain of step nodes + terminal node)
and compares structural format between prediction and reference label.

Complements BERTScore (semantic) and rule-based (accuracy) evaluations by measuring
how well the model reproduces the Socratic Q&A format structure.

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
    """One Q&A line in the Socratic format."""
    text: str
    has_question_mark: bool = False
    has_star_separator: bool = False
    has_expression: bool = False
    has_valid_expression: bool = False
    is_duplicate_question: bool = False


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

    for line in lines:
        # Check if this is the terminal line
        if re.search(r"#{3,4}\s*", line):
            after_hash = re.split(r"#{3,4}\s*", line, maxsplit=1)
            answer_part = after_hash[-1].strip() if len(after_hash) > 1 else ""
            dag.terminal = TerminalNode(
                text=line,
                has_hash_separator=True,
                has_numeric_answer=bool(_NUMERIC.search(answer_part)),
            )
            continue

        # Otherwise it's a step node
        has_q = "?" in line
        # Extract question text (everything before ?) for duplicate detection
        q_text = ""
        if has_q:
            q_text = line.split("?")[0].strip().lower()

        is_dup = False
        if q_text:
            if q_text in seen_questions:
                is_dup = True
            seen_questions.add(q_text)

        node = StepNode(
            text=line,
            has_question_mark=has_q,
            has_star_separator=bool(_STAR_SEP.search(line)),
            has_expression=bool(_EXPRESSION.search(line)),
            has_valid_expression=bool(_VALID_EXPRESSION.search(line)),
            is_duplicate_question=is_dup,
        )
        dag.steps.append(node)

    return dag


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

_STEP_ATTRS = [
    "has_question_mark",
    "has_star_separator",
    "has_expression",
    "has_valid_expression",
]


def _step_format_score(step: StepNode) -> float:
    """Fraction of boolean attributes that are True for a single step."""
    vals = [getattr(step, attr) for attr in _STEP_ATTRS]
    return sum(vals) / len(vals)


def compare_dags(pred_dag: SocraticDAG, label_dag: SocraticDAG) -> dict:
    """Compare prediction DAG against label DAG, returning all metrics."""
    pred_n = len(pred_dag.steps)
    label_n = len(label_dag.steps)
    max_n = max(pred_n, label_n, 1)

    # step_count_ratio
    step_count_ratio = min(pred_n, label_n, 1) / max_n if max_n > 0 else 0.0
    # Use actual counts (not clamped to 1) when both are nonzero
    if pred_n > 0 and label_n > 0:
        step_count_ratio = min(pred_n, label_n) / max(pred_n, label_n)

    # newline_separation_score (same formula, explicit name)
    newline_separation_score = step_count_ratio

    # format_score_mean: average format score across prediction steps
    if pred_dag.steps:
        format_scores = [_step_format_score(s) for s in pred_dag.steps]
        format_score_mean = statistics.mean(format_scores)
    else:
        format_score_mean = 0.0

    # structural_similarity: positional alignment of step attributes
    if pred_n > 0 and label_n > 0:
        aligned_len = max(pred_n, label_n)
        # Pad shorter list by repeating last step
        pred_steps = list(pred_dag.steps)
        label_steps = list(label_dag.steps)
        while len(pred_steps) < aligned_len:
            pred_steps.append(pred_steps[-1])
        while len(label_steps) < aligned_len:
            label_steps.append(label_steps[-1])

        pos_scores = []
        for p, l in zip(pred_steps, label_steps):
            matches = sum(
                getattr(p, attr) == getattr(l, attr) for attr in _STEP_ATTRS
            )
            pos_scores.append(matches / len(_STEP_ATTRS))
        structural_similarity = statistics.mean(pos_scores)
    else:
        structural_similarity = 0.0

    # duplicate_question_ratio
    if pred_dag.steps:
        dup_count = sum(1 for s in pred_dag.steps if s.is_duplicate_question)
        duplicate_question_ratio = dup_count / len(pred_dag.steps)
    else:
        duplicate_question_ratio = 0.0

    # terminal_present
    terminal_present = pred_dag.terminal is not None and pred_dag.terminal.has_hash_separator

    # dag_similarity (weighted composite)
    dag_similarity = (
        0.30 * step_count_ratio
        + 0.30 * format_score_mean
        + 0.25 * structural_similarity
        + 0.15 * (1.0 - duplicate_question_ratio)
    )

    return {
        "pred_step_count": pred_n,
        "label_step_count": label_n,
        "step_count_ratio": round(step_count_ratio, 4),
        "newline_separation_score": round(newline_separation_score, 4),
        "format_score_mean": round(format_score_mean, 4),
        "structural_similarity": round(structural_similarity, 4),
        "duplicate_question_ratio": round(duplicate_question_ratio, 4),
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
        label_text = row.get("label") or ""

        pred_dag = parse_socratic_dag(pred_text)
        label_dag = parse_socratic_dag(label_text)
        metrics = compare_dags(pred_dag, label_dag)

        rec = {"index": i, **metrics}
        if verbose:
            rec["predict"] = pred_text
            rec["label"] = label_text
        per_sample.append(rec)

    # Aggregate summary
    summary = {}
    if per_sample:
        for key in [
            "step_count_ratio",
            "newline_separation_score",
            "format_score_mean",
            "structural_similarity",
            "duplicate_question_ratio",
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
        print(f"  Step count ratio:        {summary['step_count_ratio_mean']:.4f} ± {summary['step_count_ratio_std']:.4f}")
        print(f"  Newline separation:      {summary['newline_separation_score_mean']:.4f} ± {summary['newline_separation_score_std']:.4f}")
        print(f"  Format score:            {summary['format_score_mean_mean']:.4f} ± {summary['format_score_mean_std']:.4f}")
        print(f"  Structural similarity:   {summary['structural_similarity_mean']:.4f} ± {summary['structural_similarity_std']:.4f}")
        print(f"  Duplicate question ratio: {summary['duplicate_question_ratio_mean']:.4f} ± {summary['duplicate_question_ratio_std']:.4f}")
        print(f"  DAG similarity:          {summary['dag_similarity_mean']:.4f} ± {summary['dag_similarity_std']:.4f}")
        print(f"  Terminal present:        {summary['terminal_present_pct']:.1f}%")
    else:
        print("  No valid samples")

    if verbose:
        print("\n[Per-sample details]")
        for rec in per_sample:
            print(
                f"  #{rec['index']}: steps={rec['pred_step_count']}/{rec['label_step_count']} "
                f"fmt={rec['format_score_mean']:.4f} struct={rec['structural_similarity']:.4f} "
                f"dag={rec['dag_similarity']:.4f} dup={rec['duplicate_question_ratio']:.4f} "
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
