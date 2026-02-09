# GSM8K Socratic — End-to-End 執行清單

在專案根目錄 `Fine-tunning-LLM` 下執行，並先啟動 venv：  
`source venv/bin/activate`

---

## 步驟 1：準備資料（含新 instruction）

```bash
python prepare_socratic.py
```

- **產出**：`data/custom/gsm8k_socratic_alpaca_train.jsonl`、`gsm8k_socratic_alpaca_test.jsonl`，以及更新 `dataset_info.json`
- **說明**：instruction 已與 Socratic Score 的 system prompt 對齊（問句? ** 該問的答案與推理，最後答案放 ####）

---

## 步驟 2：Fine-tune 模型

```bash
llamafactory-cli train m3_socratic_config.yaml
```

- **產出**：`outputs/gsm8k_socratic_qwen_m3/`（LoRA adapter 與 checkpoint）
- **說明**：使用 `gsm8k_socratic_mac` 訓練；若只做快速試跑，可在 yaml 內調整 `max_samples`

---

## 步驟 3：推論（兩種條件各跑一次）

**Condition 1 — Prompt-only（base 模型）：**

```bash
llamafactory-cli train m3_socratic_eval_promptonly.yaml
```

- **產出**：`outputs/gsm8k_socratic_qwen_m3_eval_promptonly/generated_predictions.jsonl`

**Condition 2 — Fine-tuned（載入 adapter）：**

```bash
llamafactory-cli train m3_socratic_eval_finetuned.yaml
```

- **產出**：`outputs/gsm8k_socratic_qwen_m3_eval_finetuned/generated_predictions.jsonl`

---

## 步驟 4：評估

**4a. 規則型指標**（evaluate_gsm8k.py 四項指標：Reasoning、Guidance Density、Socratic Format Adherence）：

```bash
python evaluate_gsm8k.py outputs/gsm8k_socratic_qwen_m3_eval_promptonly
python evaluate_gsm8k.py outputs/gsm8k_socratic_qwen_m3_eval_finetuned
```

**4b. Qwen2.5-14B 評估**（Reasoning 由 14B 從推理算答案 + Socratic Score 1–5）：

```bash
python evaluate_with_qwen14b.py outputs/gsm8k_socratic_qwen_m3_eval_promptonly \
  --model Qwen/Qwen2.5-14B-Instruct --quantize 4bit

python evaluate_with_qwen14b.py outputs/gsm8k_socratic_qwen_m3_eval_finetuned \
  --model Qwen/Qwen2.5-14B-Instruct --quantize 4bit
```

- 快速試跑可加：`--max-samples 20`
- Apple Silicon 可改用：`--backend mlx`（需 `pip install mlx mlx-lm`）
- 14B 約需 28GB+ 下載；磁碟不足可改用：`--model Qwen/Qwen2.5-7B-Instruct --quantize 4bit`

---

## 一覽表

| 步驟 | 指令 | 主要產出 |
|------|------|----------|
| 1 | `python prepare_socratic.py` | `data/custom/gsm8k_socratic_alpaca_{train,test}.jsonl` |
| 2 | `llamafactory-cli train m3_socratic_config.yaml` | `outputs/gsm8k_socratic_qwen_m3/` |
| 3a | `llamafactory-cli train m3_socratic_eval_promptonly.yaml` | `.../eval_promptonly/generated_predictions.jsonl` |
| 3b | `llamafactory-cli train m3_socratic_eval_finetuned.yaml` | `.../eval_finetuned/generated_predictions.jsonl` |
| 4a | `python evaluate_gsm8k.py outputs/...` | 終端輸出 Reasoning、Guidance Density、Socratic Format Adherence |
| 4b | `python evaluate_with_qwen14b.py outputs/...` | 終端輸出 Reasoning (14B)、Socratic Score；可加 `--output-json` 存檔 |

---

## 複製貼上版（依序執行）

```bash
cd /Users/yenchenchen/Fine-tunning-LLM
source venv/bin/activate

python prepare_socratic.py
llamafactory-cli train m3_socratic_config.yaml
llamafactory-cli train m3_socratic_eval_promptonly.yaml
llamafactory-cli train m3_socratic_eval_finetuned.yaml

python evaluate_gsm8k.py outputs/gsm8k_socratic_qwen_m3_eval_promptonly
python evaluate_gsm8k.py outputs/gsm8k_socratic_qwen_m3_eval_finetuned

python evaluate_with_qwen14b.py outputs/gsm8k_socratic_qwen_m3_eval_promptonly --model Qwen/Qwen2.5-14B-Instruct --quantize 4bit
python evaluate_with_qwen14b.py outputs/gsm8k_socratic_qwen_m3_eval_finetuned --model Qwen/Qwen2.5-14B-Instruct --quantize 4bit
```
