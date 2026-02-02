#!/usr/bin/env python3
"""
準備 GSM8K socratic 子集給 LLaMA-Factory 訓練與評估。

- Dataset: openai/gsm8k, subset = socratic
- Instruction（研究用 prompt）與 evaluate_with_qwen72b.py 的 SOCRATIC_SYSTEM 對齊：
  問句以 ? 結尾，每句後用「 ** 」接該問的答案與推理，最後答案放在 #### 後。
- 輸出 Alpaca 格式 jsonl，並更新 data/custom/dataset_info.json。
"""
import json
import os
from datasets import load_dataset
from tqdm import tqdm

# 與 Socratic Score 的 system prompt 對齊：要求「問句? ** 該問的答案與推理」，最後答案放 ####
SOCRATIC_INSTRUCTION = (
    "Use socratic questions to guide through the problem. End each guiding question with '?'. "
    "After each question, write ' ** ' then the answer and reasoning for that question "
    "(e.g. 'How many eggs? ** She sells 16-3-4=9 eggs.'). Put the final answer after ####."
)


def main():
    print("正在下載並準備 GSM8K socratic 資料集...")
    ds = load_dataset("openai/gsm8k", "socratic")

    def to_alpaca(ex):
        return {
            "instruction": SOCRATIC_INSTRUCTION,
            "input": ex["question"],
            "output": ex["answer"],
        }

    os.makedirs("data/custom", exist_ok=True)
    for split in ["train", "test"]:
        path = f"data/custom/gsm8k_socratic_alpaca_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in tqdm(ds[split], desc=f"Converting {split}"):
                f.write(json.dumps(to_alpaca(ex), ensure_ascii=False) + "\n")

    # 讀取現有 dataset_info，新增 socratic 條目（保留既有 gsm8k_mac 等）
    config_path = "data/custom/dataset_info.json"
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    dataset_info["gsm8k_socratic_mac"] = {
        "file_name": "gsm8k_socratic_alpaca_train.jsonl",
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }
    dataset_info["gsm8k_socratic_mac_test"] = {
        "file_name": "gsm8k_socratic_alpaca_test.jsonl",
        "formatting": "alpaca",
        "columns": {"prompt": "instruction", "query": "input", "response": "output"},
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    print("✅ GSM8K socratic 資料準備完成並已註冊（gsm8k_socratic_mac / gsm8k_socratic_mac_test）。")


if __name__ == "__main__":
    main()
