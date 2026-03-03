#!/usr/bin/env python3
"""
從 evaluate_with_qwen72b.py 的 --output-json 結果中，擷取 Socratic Score 最高的前 N 筆，
並顯示完整的 prompt、predict、label。
使用前須先執行：
  python evaluate_with_qwen72b.py outputs/gsm8k_socratic_qwen_eval_promptonly \\
    --model mlx-community/Qwen2.5-14B-Instruct-4bit --backend mlx \\
    -o outputs/gsm8k_socratic_qwen_eval_promptonly/eval_socratic_results.json
"""
import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json", help="Path to eval output JSON (from -o)")
    ap.add_argument("-n", "--top", type=int, default=10)
    ap.add_argument("--predictions", default=None, help="Path to generated_predictions.jsonl (default: same dir as eval_json)")
    args = ap.parse_args()

    with open(args.eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    pred_path = args.predictions
    if pred_path is None:
        pred_dir = os.path.dirname(args.eval_json)
        pred_path = os.path.join(pred_dir, "generated_predictions.jsonl")
    if not os.path.isfile(pred_path):
        pred_path = data.get("predictions_path", "")
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    rows = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    per_sample = data.get("per_sample", [])
    # 只保留有 socratic_score 的
    scored = [(r, rows[r["index"]] if r["index"] < len(rows) else None) for r in per_sample if r.get("socratic_score") is not None]
    scored.sort(key=lambda x: x[0]["socratic_score"], reverse=True)
    top = scored[: args.top]

    print(f"# Top {args.top} Prompt-only samples by Socratic Score\n")
    for rank, (rec, row) in enumerate(top, 1):
        score = rec["socratic_score"]
        idx = rec["index"]
        prompt = (row or {}).get("prompt", "")
        predict = (row or {}).get("predict", "")
        label = (row or {}).get("label", "")
        gt = rec.get("gt", "")

        print("=" * 80)
        print(f"## #{rank}  Index={idx}  Socratic Score={score}  GT={gt}")
        print("-" * 40)
        print("### Prompt (instruction + question):")
        print(prompt[:600] + ("..." if len(prompt) > 600 else ""))
        print()
        print("### Model prediction:")
        print(predict[:800] + ("..." if len(predict) > 800 else ""))
        print()
        print("### Label (ground truth):")
        print(label[:400] + ("..." if len(label) > 400 else ""))
        print()


if __name__ == "__main__":
    main()
