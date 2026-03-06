#!/usr/bin/env python3
"""
Significance tests for Fine-tuned vs Prompt-only (same GSM8K test set).

- **Accuracy**: two-proportion z-test (binary correct/incorrect per sample).
- **Format Adherence (DAG)** and **BERTScore (F1)**: paired t-test (per-sample scores).

Reads eval_rule_results.json for accuracy; eval_dag_results.json and eval_bert_results.json
for DAG and BERT (run evaluate_dag.py and evaluate_bert.py on both eval dirs first).

Usage:
  # Accuracy only (two JSON paths or document proportions)
  python scripts/significance_accuracy.py <path_finetuned_rule.json> <path_promptonly_rule.json>
  python scripts/significance_accuracy.py --p1 0.6937 --p2 0.6384 --n 1319

  # All metrics (two eval dirs; looks for eval_rule_results.json, eval_dag_results.json, eval_bert_results.json)
  python scripts/significance_accuracy.py --dirs <finetuned_dir> <promptonly_dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os


def two_proportion_ztest(
    count1: int, n1: int, count2: int, n2: int
) -> tuple[float, float]:
    """Two-proportion z-test (pooled proportion). Returns (z_stat, p_value_two_tailed)."""
    p1 = count1 / n1
    p2 = count2 / n2
    p_pool = (count1 + count2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se <= 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    # Two-tailed p-value: 2 * (1 - Phi(|z|)), Phi = standard normal CDF
    from math import erf, sqrt
    phi_z = 0.5 * (1 + erf(abs(z) / sqrt(2)))
    p_value = 2 * (1 - phi_z)
    return (z, max(0.0, min(1.0, p_value)))


def paired_ttest(scores1: list[float], scores2: list[float]) -> tuple[float, float, float]:
    """Paired t-test. Returns (t_stat, p_value_two_tailed, mean_diff). Assumes same order (same n)."""
    n = min(len(scores1), len(scores2))
    if n < 2:
        return (0.0, 1.0, 0.0)
    diffs = [scores1[i] - scores2[i] for i in range(n)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
    std_d = math.sqrt(var_d)
    if std_d <= 0:
        return (0.0, 1.0, mean_d)
    t = mean_d / (std_d / math.sqrt(n))
    # Two-tailed p from t-distribution (df=n-1). For large n, t ~ N(0,1).
    from math import erf, sqrt
    phi_t = 0.5 * (1 + erf(abs(t) / sqrt(2)))
    p_value = 2 * (1 - phi_t)
    return (t, max(0.0, min(1.0, p_value)), mean_d)


def load_counts_from_json(path: str) -> tuple[int, int]:
    """Return (n_correct, n_total) from eval_rule_results.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = data.get("n_samples", 0)
    per = data.get("per_sample", [])
    if per and n:
        correct = sum(1 for s in per if s.get("reasoning_correct_rule") is True)
        return (correct, n)
    acc = data.get("summary", {}).get("reasoning_accuracy_rule", 0)
    if n and acc is not None:
        count = round(n * acc / 100.0)
        return (count, n)
    return (0, 0)


def load_paired_scores(path1: str, path2: str, key: str, index_key: str = "index") -> tuple[list[float], list[float], int] | None:
    """Load per-sample scores from two JSON files; align by index. Returns (scores1, scores2, n) or None."""
    def load_key(path: str) -> dict[int, float] | None:
        if not path or not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        per = data.get("per_sample", [])
        if not per:
            return None
        return {s.get(index_key, i): s[key] for i, s in enumerate(per) if key in s}
    m1 = load_key(path1)
    m2 = load_key(path2)
    if m1 is None or m2 is None:
        return None
    indices = sorted(set(m1) & set(m2))
    if not indices:
        return None
    return ([m1[i] for i in indices], [m2[i] for i in indices], len(indices))


def run_accuracy_test(
    count1: int, n1: int, count2: int, n2: int,
    label1: str = "Fine-tuned", label2: str = "Prompt-only",
) -> None:
    """Run and print two-proportion z-test for accuracy."""
    p1 = count1 / n1
    p2 = count2 / n2
    z, p_value = two_proportion_ztest(count1, n1, count2, n2)
    print("--- Reasoning Accuracy (two-proportion z-test) ---")
    print(f"  {label1}: {count1}/{n1} = {100*p1:.2f}%")
    print(f"  {label2}: {count2}/{n2} = {100*p2:.2f}%")
    print(f"  z = {z:.4f}, p (two-tailed) = {p_value:.4f}")
    if p_value < 0.05:
        print("  Conclusion: Difference is statistically significant (p < 0.05).")
    else:
        print("  Conclusion: Difference is not statistically significant (p >= 0.05).")
    print()


def run_paired_test(
    scores1: list[float], scores2: list[float], n: int,
    metric_name: str, label1: str = "Fine-tuned", label2: str = "Prompt-only",
) -> None:
    """Run and print paired t-test for a continuous metric."""
    mean1 = sum(scores1) / n
    mean2 = sum(scores2) / n
    t, p_value, mean_diff = paired_ttest(scores1, scores2)
    print(f"--- {metric_name} (paired t-test, n={n}) ---")
    print(f"  {label1}: mean = {mean1:.4f}")
    print(f"  {label2}: mean = {mean2:.4f}")
    print(f"  mean difference = {mean_diff:.4f}, t = {t:.4f}, p (two-tailed) = {p_value:.4f}")
    if p_value < 0.05:
        print("  Conclusion: Difference is statistically significant (p < 0.05).")
    else:
        print("  Conclusion: Difference is not statistically significant (p >= 0.05).")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Significance tests: Accuracy (z-test), Format Adherence & BERTScore (paired t-test)."
    )
    parser.add_argument(
        "finetuned_json",
        nargs="?",
        help="Path to eval_rule_results.json for Fine-tuned (or first eval dir when using --dirs).",
    )
    parser.add_argument(
        "promptonly_json",
        nargs="?",
        help="Path to eval_rule_results.json for Prompt-only (or second eval dir when using --dirs).",
    )
    parser.add_argument("--p1", type=float, help="Proportion 1 (Fine-tuned), e.g. 0.6937")
    parser.add_argument("--p2", type=float, help="Proportion 2 (Prompt-only), e.g. 0.6384")
    parser.add_argument("--n", type=int, help="Sample size (same for both), e.g. 1319")
    parser.add_argument(
        "--dirs",
        nargs=2,
        metavar=("FINETUNED_DIR", "PROMPTONLY_DIR"),
        help="Run all metrics: provide two eval dirs (each with eval_rule_results.json; optional eval_dag_results.json, eval_bert_results.json).",
    )
    args = parser.parse_args()

    if args.dirs:
        fd, pd = args.dirs[0], args.dirs[1]
        rule_f = os.path.join(fd, "eval_rule_results.json")
        rule_p = os.path.join(pd, "eval_rule_results.json")
        dag_f = os.path.join(fd, "eval_dag_results.json")
        dag_p = os.path.join(pd, "eval_dag_results.json")
        bert_f = os.path.join(fd, "eval_bert_results.json")
        bert_p = os.path.join(pd, "eval_bert_results.json")

        # Accuracy
        if os.path.isfile(rule_f) and os.path.isfile(rule_p):
            count1, n1 = load_counts_from_json(rule_f)
            count2, n2 = load_counts_from_json(rule_p)
            n1 = n2 = min(n1, n2)
            run_accuracy_test(count1, n1, count2, n2)
        else:
            print("Accuracy: eval_rule_results.json not found in both dirs. Run evaluate_rule_based.py first.\n")

        # Format Adherence (DAG)
        pair = load_paired_scores(dag_f, dag_p, "format_score_mean")
        if pair is not None:
            s1, s2, n = pair
            run_paired_test(s1, s2, n, "Format Adherence (DAG format_score_mean)")
        else:
            print("Format Adherence: eval_dag_results.json not found or missing per_sample. Run evaluate_dag.py on both eval dirs first.\n")

        # BERTScore F1
        pair = load_paired_scores(bert_f, bert_p, "bertscore_f1")
        if pair is not None:
            s1, s2, n = pair
            run_paired_test(s1, s2, n, "BERTScore F1")
        else:
            print("BERTScore: eval_bert_results.json not found or missing per_sample. Run evaluate_bert.py on both eval dirs first.\n")
        return

    if args.p1 is not None and args.p2 is not None and args.n is not None:
        n1 = n2 = args.n
        count1 = round(args.p1 * args.n)
        count2 = round(args.p2 * args.n)
        count1 = max(0, min(args.n, count1))
        count2 = max(0, min(args.n, count2))
        label1, label2 = "Fine-tuned", "Prompt-only"
    elif args.finetuned_json and args.promptonly_json:
        count1, n1 = load_counts_from_json(args.finetuned_json)
        count2, n2 = load_counts_from_json(args.promptonly_json)
        label1, label2 = "Fine-tuned", "Prompt-only"
        if n1 != n2:
            print("Warning: n1 != n2; using common n = min(n1,n2) for pooled SE.")
            n1 = n2 = min(n1, n2)
    else:
        parser.error("Either provide two JSON paths, --p1/--p2/--n, or --dirs <finetuned_dir> <promptonly_dir>.")
        return

    run_accuracy_test(count1, n1, count2, n2, label1, label2)
    p1 = count1 / n1
    p2 = count2 / n2
    z, pv = two_proportion_ztest(count1, n1, count2, n2)
    print(f"n_samples={n1} p1={p1:.4f} p2={p2:.4f} z={z:.4f} p_value={pv:.4f}")


if __name__ == "__main__":
    main()
