import os
import json
import yaml
import subprocess
import sys
from datasets import load_dataset
from tqdm import tqdm

# --- [1] 資料準備：GSM8K 轉 Alpaca ---
def prepare_data():
    print("正在下載並準備 GSM8K 資料集...")
    ds = load_dataset("openai/gsm8k", "main")
    
    def to_alpaca(ex):
        return {
            "instruction": "Solve the math word problem step by step, and give the final answer.",
            "input": ex["question"],
            "output": ex["answer"]
        }
    
    os.makedirs("data/custom", exist_ok=True)
    for split in ["train", "test"]:
        path = f"data/custom/gsm8k_alpaca_{split}.jsonl"
        # 檢查檔案是否已存在，避免重複轉換
        with open(path, "w", encoding="utf-8") as f:
            for ex in tqdm(ds[split], desc=f"Converting {split}"):
                f.write(json.dumps(to_alpaca(ex), ensure_ascii=False) + "\n")

    # 註冊資料集索引 [cite: 842-853]
    # file_name 為相對於 dataset_dir 的路徑，勿重複 "custom/"
    dataset_info = {
        "gsm8k_mac": {
            "file_name": "gsm8k_alpaca_train.jsonl",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"}
        },
        "gsm8k_mac_test": {
            "file_name": "gsm8k_alpaca_test.jsonl",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"}
        }
    }
    with open("data/custom/dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print("✅ 資料準備完成並已註冊。")

# --- [2] 啟動 M3 優化訓練 ---
def run_training():
    config = {
        "model_name_or_path": "Qwen/Qwen2.5-3B-Instruct",
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "dataset": "gsm8k_mac",
        "dataset_dir": "data/custom",
        "template": "qwen",
        "cutoff_len": 512,
        "max_samples": 500,  
        "output_dir": "outputs/gsm8k_qwen_m3",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2.0e-4,
        "num_train_epochs": 1.0,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": True,         # M3 支援混合精度 [cite: 137]
        ##"device_map": "auto", 
        "logging_steps": 5,   # 縮短日誌步數，更快看到進度 
        "save_steps": 100,
        "plot_loss": True,     # 訓練完會自動畫 Loss 圖 
        "include_num_input_tokens_seen": True # 顯示已處理的 token 數
    }
    
    with open("m3_config.yaml", "w") as f:
        yaml.dump(config, f)
    
    print("🚀 啟動 LLaMA-Factory 訓練流程...")
    
    # 使用 subprocess 確保 stdout 即時輸出到終端機
    try:
        # llamafactory-cli 會自動偵測 MPS 並印出進度條
        subprocess.run(
            ["llamafactory-cli", "train", "m3_config.yaml"],
            check=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ 訓練中斷或出錯: {e}")
    except FileNotFoundError:
        print("❌ 找不到 llamafactory-cli，請確認是否已執行 pip install -e .")

if __name__ == "__main__":
    prepare_data()
    run_training()