# GSM8K Socratic 研究 — 下一步

依你目前架構（LlamaFactory + `openai/gsm8k` socratic + Qwen2.5-3B-Instruct，M3 Mac 本機），已完成：

- **資料**：`prepare_socratic_dataset.py` 下載 socratic 子集，轉成 Alpaca，註冊 `gsm8k_socratic_train` / `gsm8k_socratic_test`
- **訓練**：`socratic_train_config.yaml`（SFT on socratic，與研究用 prompt 一致）
- **評估**：`socratic_finetuned_eval.yaml`（微調後 adapter）、`socratic_promptonly_eval.yaml`（僅 base 模型，相同 prompt）
- **指標**：`evaluate_rule_based.py` 支援四項指標（Reasoning、Guidance Density、Socratic Format Adherence）；**Qwen2.5-14B** 評估使用 `evaluate_with_qwen14b.py`（Reasoning 由 14B 從推理算出答案＋Socratic Score）

---

## 1. 環境與路徑

- 在 **專案根目錄** `Fine-tunning-LLM` 下執行所有指令。
- 使用已安裝 LlamaFactory 的 venv：  
  `source venv/bin/activate` 或直接  
  `venv/bin/python` / `venv/bin/llamafactory-cli`。

---

## 2. 依序執行

### 2.1 準備 Socratic 資料（只需跑一次）

```bash
python prepare_socratic_dataset.py
```

會產生 `data/custom/gsm8k_socratic_alpaca_{train,test}.jsonl` 並更新 `data/custom/dataset_info.json`。

### 2.2 訓練（Condition 2：Fine-tuned）

```bash
llamafactory-cli train socratic_train_config.yaml
```

輸出目錄：`outputs/gsm8k_socratic_qwen`。

### 2.3 推論

**Condition 1 — Prompt-only（base 模型，相同 prompt）：**

```bash
llamafactory-cli train socratic_promptonly_eval.yaml
```

輸出：`outputs/gsm8k_socratic_qwen_eval_promptonly/generated_predictions.jsonl`。

**Condition 2 — Fine-tuned（載入 socratic 微調 adapter）：**

```bash
llamafactory-cli train socratic_finetuned_eval.yaml
```

輸出：`outputs/gsm8k_socratic_qwen_eval_finetuned/generated_predictions.jsonl`。

### 2.4 評估

**規則型指標**（`evaluate_rule_based.py`）：

```bash
python evaluate_rule_based.py outputs/gsm8k_socratic_qwen_eval_promptonly
python evaluate_rule_based.py outputs/gsm8k_socratic_qwen_eval_finetuned
```

會輸出：Reasoning (Accuracy Pass@1)、Guidance Density、Socratic Format Adherence。

**Qwen2.5-14B 評估**（Reasoning 由 14B 從「拿掉 #### 的推理」算出答案；Socratic Score 由 14B 評 1–5）：

```bash
# 4-bit 量化，快速試跑可加 --max-samples 20
python evaluate_with_qwen14b.py outputs/gsm8k_socratic_qwen_eval_promptonly \
  --model Qwen/Qwen2.5-14B-Instruct --quantize 4bit
python evaluate_with_qwen14b.py outputs/gsm8k_socratic_qwen_eval_finetuned \
  --model Qwen/Qwen2.5-14B-Instruct --quantize 4bit

# Apple Silicon 可改用 MLX（需 pip install mlx mlx-lm）
python evaluate_with_qwen14b.py outputs/... --backend mlx
```

依賴：`transformers`、`torch`；4-bit 時 `pip install bitsandbytes`。可設 `EVAL_MODEL_ID`、`EVAL_QUANTIZE`、`EVAL_BACKEND` 環境變數覆寫預設。

**注意**：14B 約需 **28GB+ 下載**。若磁碟不足可：
- **改用較小評估模型**：  
  `python evaluate_with_qwen14b.py ... --model Qwen/Qwen2.5-7B-Instruct --quantize 4bit`
- **改用 MLX**（Apple Silicon）：  
  `pip install mlx mlx-lm` 後  
  `python evaluate_with_qwen14b.py ... --backend mlx`
- 或先只用 `evaluate_rule_based.py` 的規則型指標，暫不做 14B 評估。

---

## 3. 研究用 prompt（已包在資料裡）

- Instruction：  
  `Use socratic question to guide me through the question to answer. End each guiding question with '?'. Put the final answer after ####.`
- 訓練與評估的 eval 資料皆使用此 instruction，因此兩條件使用 **相同 prompt**。

---

## 4. 評估維度與後續

| 維度 | 指標 | 說明 |
|------|------|------|
| Reasoning | Accuracy (Pass@1) | `evaluate_rule_based.py`（#### 答案比對）；**Qwen2.5-14B**：`evaluate_with_qwen14b.py`（拿掉 #### 後由 14B 算出答案再比對） |
| Guidance Density | Question count | 每則回覆 `?` 的數量，`evaluate_rule_based.py` 已輸出 |
| Socratic Format Adherence | 格式遵從 | 是否符合 fine-tuned 輸出格式（Question? ** <<expr>>，####），`evaluate_rule_based.py` 已實作 |
| Guidance Quality | Socratic Score | `evaluate_with_qwen14b.py` 使用 **Qwen2.5-14B** 以 Prompt 評「啟發性」v.s.「告知性」（1–5） |

---

## 5. 與目前 main 實驗的差異

- 你先前用的是 **main** 子集與 `gsm8k_mac`，且不同 instruction。
- 現在改為 **socratic** 子集、`gsm8k_socratic_train`，以及上述研究用 prompt；訓練與評估都用 socratic，以對齊研究設計。

若之後要**快速試跑**（例如除錯），可在對應 yaml 裡加上 `max_samples: 50` 等限制，再跑 `train` / `eval`。

---

## 6. 若評估方式有更改：LLM 的 prompt 要改哪裡？

| 用途 | 對應位置 | 說明 |
|------|----------|------|
| **生成用 prompt**（3B 推論時看到的 instruction + 題目） | `data/custom/gsm8k_socratic_alpaca_*.jsonl` 的 `instruction` 欄位；重新產生資料時則改 `prepare_socratic_dataset.py` 的 `SOCRATIC_INSTRUCTION` | 與訓練資料一致，答案格式請用 `####`（評估腳本已同時支援 `###` 與 `####`） |
| **14B Reasoning**（由推理文算出數字答案） | `evaluate_with_qwen14b.py` 的 `REASONING_SYSTEM`、`REASONING_USER` | 若改成「只比對 #### 後答案」或不同題目，在此改 14B 的系統/使用者提示 |
| **14B Socratic Score**（1–5 評分準則） | `evaluate_with_qwen14b.py` 的 `SOCRATIC_SYSTEM`、`SOCRATIC_USER` | 若評分維度或尺度改變（例如改為「啟發性 vs 告知性」的定義），在此改 14B 的評分 prompt |

規則型指標（Accuracy 從 ####/### 取答案、Question count、Socratic Format Adherence）在 `evaluate_rule_based.py`，邏輯與 prompt 無關；若只改「是否用 14B 算答案 / 是否用 14B 評 Socratic」，只需改上述 14B 的常數即可。
