#!/usr/bin/env python3
"""
使用 Qwen2.5-14B 作為評估模型，對 generated_predictions.jsonl 進行：

1. Reasoning (Accuracy via 14B)：拿掉 #### 後的推理內容交給 14B，請其給出最終數字答案，再與 GT 比對。
2. Socratic Score：以 Prompt 請 14B 評估回覆的「啟發性」v.s.「告知性」（1–5）。

依賴：transformers, torch；4-bit 時需 bitsandbytes。Apple Silicon 可選 mlx / mlx_lm 搭配
      mlx-community/Qwen2.5-14B-Instruct-4bit（--backend mlx）。

用法：
  python evaluate_with_qwen14b.py outputs/gsm8k_socratic_qwen_m3_eval_finetuned \\
    --model Qwen/Qwen2.5-14B-Instruct --quantize 4bit --max-samples 20
  python evaluate_with_qwen14b.py path/to/generated_predictions.jsonl --backend mlx
  # 若中斷（關機/當機），加上 --resume 可從上次進度繼續
  python evaluate_with_qwen14b.py outputs/... -o outputs/.../eval_full_results.json --resume
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

# Reuse format-free extraction from evaluate_gsm8k
from evaluate_gsm8k import extract_gsm8k_answer as _extract_gsm8k_answer, normalize_answer


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract the final answer (format-free). Delegates to evaluate_gsm8k."""
    ans, _ = _extract_gsm8k_answer(text)
    return ans


def reasoning_only(predict: str) -> str:
    """Remove ####/### and the final answer; return reasoning part only."""
    if not predict:
        return ""
    for sep in ("####", "###"):
        if sep in predict:
            parts = re.split(r"\s*" + re.escape(sep) + r"\s*", predict)
            return parts[0].strip()
    return predict


def normalize_number(s: str) -> str:
    """Remove commas, strip."""
    return (s or "").replace(",", "").strip()


def parse_first_number(text: str) -> str | None:
    """Extract first number (including decimals) from LLM reply. Prefer last number if multiple."""
    if not text:
        return None
    # Match integers or decimals, possibly with commas
    candidates = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not candidates:
        return None
    return normalize_number(candidates[-1])


def parse_socratic_score(text: str) -> int | None:
    """Expect 1–5 from LLM. Return None if unparseable."""
    if not text:
        return None
    m = re.search(r"[1-5]", text)
    return int(m.group()) if m else None


def count_questions(text: str) -> int:
    return (text or "").count("?")


# --- Prompts for 14B（若評估方式有更改，請在此修改 REASONING_* / SOCRATIC_* 以對齊新準則）---
REASONING_SYSTEM = (
    "You are an expert grader. Given a model's step-by-step reasoning for a math word problem, "
    "you must determine the final numeric answer. Reply with only that number, nothing else."
)
REASONING_USER = (
    "The model's reasoning (the final answer has been removed) is:\n\n{reasoning}\n\n"
    "What is the final numeric answer? Reply with only the number."
)
# 與訓練資料對齊：gsm8k socratic 格式為「問句? ** 該問題的答案（含推理）」，屬 scaffolding，非純蘇格拉底
SOCRATIC_SYSTEM = (
    "You evaluate whether a math tutoring response follows the Socratic-style scaffolding format: "
    "using guiding questions (each ending with ?) to break down the problem, with the answer to each question attached after '**'. "
    "The expected format is 'Question? ** answer and reasoning for that question' (e.g. 'How many eggs? ** She sells 16-3-4=9 eggs.'). "
    "Rate 1–5: 1 = no guiding questions, just equations or direct solution; "
    "3 = some questions but missing the 'Question? ** answer' structure; "
    "5 = clearly uses multiple guiding questions, each followed by '**' and the answer/reasoning for that question. "
    "Reply with only one digit."
)
SOCRATIC_USER = "Rate this response (1–5):\n\n{response}\n\nReply with only one digit."


def load_model_transformers(model_id: str, quantize: str, device_map: str = "auto"):
    """Load Qwen2.5-14B via transformers. Optionally 4bit/8bit via bitsandbytes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs = {"trust_remote_code": True, "device_map": device_map}
    if quantize == "4bit":
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        except Exception as e:
            raise RuntimeError(
                "4-bit requires bitsandbytes. Install: pip install bitsandbytes"
            ) from e
    elif quantize == "8bit":
        try:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except Exception as e:
            raise RuntimeError(
                "8-bit requires bitsandbytes. Install: pip install bitsandbytes"
            ) from e
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return model, tok


def load_model_mlx(model_id: str = "mlx-community/Qwen2.5-14B-Instruct-4bit"):
    """Load MLX quantized Qwen2.5-14B for Apple Silicon."""
    try:
        import mlx_lm
    except ImportError as e:
        raise RuntimeError(
            "MLX backend requires: pip install mlx mlx-lm"
        ) from e
    model, tok = mlx_lm.load(model_id)
    return model, tok


def _model_device(model):
    if getattr(model, "device", None) is not None:
        return model.device
    return next(model.parameters()).device


def generate_transformers(model, tokenizer, messages: list[dict], max_new_tokens: int = 64):
    """Chat-style generate with transformers."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    dev = _model_device(model)
    inputs = tokenizer(text, return_tensors="pt").to(dev)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return reply.strip()


def generate_mlx(model, tokenizer, messages: list[dict], max_tokens: int = 64):
    """Chat-style generate with MLX."""
    import mlx_lm

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    reply = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens)
    return reply.strip()


def run_eval(
    predictions_path: str,
    model_id: str = "Qwen/Qwen2.5-14B-Instruct",
    backend: str = "transformers",
    quantize: str = "4bit",
    max_samples: int | None = None,
    metrics: str = "both",
    output_json: str | None = None,
    verbose: bool = False,
    dry_run: bool = False,
    resume: bool = False,
    checkpoint_every: int = 50,
) -> None:
    if os.path.isdir(predictions_path):
        predictions_path = os.path.join(predictions_path, "generated_predictions.jsonl")
    if not os.path.isfile(predictions_path):
        raise FileNotFoundError(f"Not found: {predictions_path}")

    rows = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if max_samples is not None:
        rows = rows[: max_samples]
    n = len(rows)
    print(f"Loaded {n} predictions from {predictions_path}")
    print(f"Evaluator: {model_id} (backend={backend}, quantize={quantize})")
    print()

    do_reasoning = metrics in ("reasoning", "both")
    do_socratic = metrics in ("socratic", "both")

    if dry_run:
        print("[dry-run] Skipping model load. First reasoning prompt (excerpt):")
        r0 = reasoning_only(rows[0].get("predict") or "") if rows else ""
        u = REASONING_USER.format(reasoning=(r0[:400] + "..." if len(r0) > 400 else r0))
        print(u[:500] + "..." if len(u) > 500 else u)
        print("\n[dry-run] Done.")
        return

    # Load model
    if backend == "mlx":
        model, tokenizer = load_model_mlx(
            model_id if "mlx" in model_id else "mlx-community/Qwen2.5-14B-Instruct-4bit"
        )
        def generate(messages):
            return generate_mlx(model, tokenizer, messages)
    else:
        model, tokenizer = load_model_transformers(model_id, quantize)
        def generate(messages):
            return generate_transformers(model, tokenizer, messages)

    correct_llm = 0
    total_reasoning = 0
    socratic_scores: list[int] = []
    per_sample = []
    existing_by_index: dict[int, dict] = {}

    # Resume: load existing results if output_json exists and has LLM-evaluated samples
    if resume and output_json and os.path.isfile(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            for r in loaded.get("per_sample", []):
                idx = r.get("index")
                if idx is not None and (r.get("reasoning_llm_raw") is not None or r.get("socratic_llm_raw") is not None):
                    existing_by_index[idx] = r
            if existing_by_index:
                print(f"Resuming: found {len(existing_by_index)} already-evaluated samples")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Resume: could not load {output_json}: {e}. Starting fresh.")

    def save_checkpoint():
        if not output_json:
            return
        rule_correct = sum(1 for r in per_sample if r.get("reasoning_correct_rule"))
        rule_total = len(per_sample)
        qc_vals = [r["question_count"] for r in per_sample]
        fmt_ok = sum(1 for r in per_sample if r.get("format_adherence"))
        socratic_vals = [r["socratic_score"] for r in per_sample if r.get("socratic_score") is not None]
        out = {
            "predictions_path": predictions_path,
            "model_id": model_id,
            "backend": backend,
            "quantize": quantize,
            "n_samples": n,
            "summary": {
                "reasoning_accuracy_rule": 100.0 * rule_correct / rule_total if rule_total else None,
                "reasoning_accuracy_llm": 100.0 * correct_llm / total_reasoning if total_reasoning else None,
                "question_count_mean": sum(qc_vals) / len(qc_vals) if qc_vals else None,
                "format_adherence_pct": 100.0 * fmt_ok / rule_total if rule_total else None,
                "socratic_score_mean": sum(socratic_vals) / len(socratic_vals) if socratic_vals else None,
            },
            "per_sample": per_sample,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    for i, row in enumerate(rows):
        label = row.get("label") or ""
        pred = row.get("predict") or ""
        prompt = row.get("prompt", "")
        gt = extract_gsm8k_answer(label)
        if gt is None:
            continue

        # Reuse existing result if resuming
        if i in existing_by_index:
            rec = existing_by_index[i]
            per_sample.append(rec)
            if rec.get("reasoning_llm_answer") is not None:
                total_reasoning += 1
                if rec.get("reasoning_correct_llm"):
                    correct_llm += 1
            if rec.get("socratic_score") is not None:
                socratic_scores.append(rec["socratic_score"])
            if verbose:
                print(f"  [{i+1}/{n}] (resumed) gt={gt} llm_answer={rec.get('reasoning_llm_answer')} socratic={rec.get('socratic_score')}")
            continue

        ans_rule = extract_gsm8k_answer(pred)
        rule_correct = ans_rule is not None and normalize_answer(gt) == normalize_answer(ans_rule)
        qc = count_questions(pred)
        format_ok = ans_rule is not None

        rec = {
            "index": i,
            "gt": gt,
            "reasoning_correct_rule": rule_correct,
            "question_count": qc,
            "format_adherence": format_ok,
            "prompt": prompt,
            "predict": pred,
            "label": label,
        }

        if do_reasoning:
            reasoning = reasoning_only(pred)
            messages = [
                {"role": "system", "content": REASONING_SYSTEM},
                {"role": "user", "content": REASONING_USER.format(reasoning=reasoning)},
            ]
            reply = generate(messages)
            ans_llm = parse_first_number(reply)
            rec["reasoning_llm_raw"] = reply
            rec["reasoning_llm_answer"] = ans_llm
            if ans_llm is not None:
                total_reasoning += 1
                if normalize_answer(gt) == normalize_answer(ans_llm):
                    correct_llm += 1
                    rec["reasoning_correct_llm"] = True
                else:
                    rec["reasoning_correct_llm"] = False

        if do_socratic:
            messages = [
                {"role": "system", "content": SOCRATIC_SYSTEM},
                {"role": "user", "content": SOCRATIC_USER.format(response=pred)},
            ]
            reply = generate(messages)
            score = parse_socratic_score(reply)
            rec["socratic_llm_raw"] = reply
            rec["socratic_score"] = score
            if score is not None:
                socratic_scores.append(score)
        else:
            rec["socratic_score"] = None
        if "reasoning_correct_llm" not in rec:
            rec["reasoning_correct_llm"] = None

        per_sample.append(rec)
        if verbose:
            print(f"  [{i+1}/{n}] gt={gt} llm_answer={rec.get('reasoning_llm_answer')} socratic={rec.get('socratic_score')}")

        # Periodic checkpoint (in case of crash)
        if output_json and len(per_sample) % checkpoint_every == 0:
            save_checkpoint()
            print(f"  Checkpoint: saved {len(per_sample)} samples")

    # Report
    print("[Qwen2.5-14B Evaluation]")
    if do_reasoning and total_reasoning:
        acc = 100.0 * correct_llm / total_reasoning
        print(f"  Reasoning (14B): {correct_llm}/{total_reasoning} = {acc:.2f}%")
    elif do_reasoning:
        print("  Reasoning (14B): N/A (no valid 14B answers)")
    if do_socratic and socratic_scores:
        avg = sum(socratic_scores) / len(socratic_scores)
        print(f"  Socratic Score (14B): mean = {avg:.2f}, min = {min(socratic_scores)}, max = {max(socratic_scores)}")
    elif do_socratic:
        print("  Socratic Score (14B): N/A (no parseable scores)")

    if output_json:
        save_checkpoint()
        print(f"\nPer-sample results (all 4 dimensions) written to {output_json}")


def main():
    import os as _os

    default_model = _os.getenv("EVAL_MODEL_ID", "Qwen/Qwen2.5-14B-Instruct")
    default_quantize = _os.getenv("EVAL_QUANTIZE", "4bit")
    default_backend = _os.getenv("EVAL_BACKEND", "transformers")

    ap = argparse.ArgumentParser(
        description="Evaluate generated_predictions.jsonl using Qwen2.5-14B (reasoning + Socratic score)."
    )
    ap.add_argument(
        "path",
        nargs="?",
        default="outputs/gsm8k_socratic_qwen_m3_eval_finetuned",
        help="Path to eval output dir or generated_predictions.jsonl",
    )
    ap.add_argument("--model", default=default_model, help="HuggingFace model id (or MLX model if --backend mlx). Env: EVAL_MODEL_ID")
    ap.add_argument("--backend", choices=("transformers", "mlx"), default=default_backend, help="Env: EVAL_BACKEND")
    ap.add_argument("--quantize", choices=("4bit", "8bit", "none"), default=default_quantize, help="Env: EVAL_QUANTIZE")
    ap.add_argument("--max-samples", type=int, default=None, help="Cap samples for quick test")
    ap.add_argument("--metrics", choices=("reasoning", "socratic", "both"), default="both")
    ap.add_argument("--output-json", "-o", default=None, help="Write per-sample results to JSON (default: {path}/eval_full_results.json)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-n", "--dry-run", action="store_true", help="Validate inputs and print first prompt only, no model load")
    ap.add_argument("--resume", action="store_true", help="Resume from existing output_json (skip already-evaluated samples)")
    ap.add_argument("--checkpoint-every", type=int, default=50, help="Save checkpoint every N samples (default: 50)")
    args = ap.parse_args()

    quantize = "none" if args.backend == "mlx" else args.quantize
    output_json = args.output_json
    if output_json is None and not args.dry_run:
        p = args.path
        if os.path.isdir(p):
            output_json = os.path.join(p, "eval_full_results.json")
        elif os.path.isfile(p):
            output_json = os.path.join(os.path.dirname(p), "eval_full_results.json")
    run_eval(
        args.path,
        model_id=args.model,
        backend=args.backend,
        quantize=quantize,
        max_samples=args.max_samples,
        metrics=args.metrics,
        output_json=output_json,
        verbose=args.verbose,
        dry_run=args.dry_run,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
