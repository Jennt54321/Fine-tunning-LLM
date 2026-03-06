# 研究設計與評估紀錄

## 1. 問題定義

在均一平台發現學生較少與模型互動，多數直接傳題目要求模型給答案。因此思考：若不以訓練學生學 prompt engineering 為主，而是**訓練模型使其回覆更具引導性**，或更能符合學生需求。

本研究以 **GSM8K**（socratic 子集）為資料集、**Qwen2.5-3B-Instruct** 為基礎模型，透過 Socratic 風格的 SFT，希望模型在回答數學問題時更懂得**引導思考**，並遵守可解析的輸出格式。

**研究範圍與限制**  
GSM8K 的設計目的是訓練模型「回答」數學問題，其 Socratic 格式著重在**拆解問題**──將解題過程分成逐步、邏輯清楚的步驟，最終得出數值答案。本計畫原本更希望模型能產生**引導性問題**（探索式、家教式的蘇格拉底對話），但微調後的輸出未達理想中的蘇格拉底品質，因此本研究未採用 LLM 打分的 Socratic 品質指標，假說與評估也隨之調整（見下方研究假說與評估指標）。目前評估仍提供規則式、DAG 結構檢查、BERTScore 及可選用的 LLM 評分等多元指標供參考。

---

## 2. 研究假說

| 代號 | 假說 |
|------|------|
| **H1** | 經 Socratic SFT 微調後的 Qwen2.5-3B，在回答數學問題時是否具有更好的**回答品質**（含蘇格拉底式引導）？ **［未驗證／資料集不符］** |
| **H2** | 微調後的模型在回答數學問題時，是否能遵守**既定輸出規範**，以利後續 parsing 與自動評估？ |
| **H3** | 微調後的模型是否能提供**更可預測的解法**（例如一致的多步驟格式、正確率提升）？ |

---

## 3. 實驗設計概要

- **資料**：`openai/gsm8k` 的 socratic 子集並轉成 Alpaca 格式。
- **訓練**：LoRA SFT。
- **對照**：**Prompt-only**（僅 base 模型 + 相同 prompt）vs **Fine-tuned**（載入上述 adapter）。
- **評估**：見 §4 評估指標與工具。

---

## 3.1 實驗環境與可重現性


### LoRA 超參數

| 參數 | 值 |
|------|----|
| `r` (rank) | 8 |
| `lora_alpha` | 16 |
| `lora_dropout` | 0.0 |
| `bias` | none |
| `target_modules` | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| `use_rslora` | false |
| `use_dora` | false |
| Base model | Qwen/Qwen2.5-3B-Instruct |

### 訓練超參數（SFT）

| 參數 | 值 |
|------|----|
| `learning_rate` | 2e-4（0.0002） |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 8（有效 batch size = 8） |
| `lr_scheduler_type` | cosine |
| `warmup_ratio` | 0.03 |

### 訓練結果摘要

| 指標 | 值 |
|------|----|
| Epoch | 1.0 |
| Train Loss | 0.2907 |
| Train Runtime | 6954s（約 1.93 小時） |
| Samples/sec | 1.075 |

---

## 4. 評估指標與工具

| 指標 | 腳本 | 說明 |
|------|------|------|
| **Reasoning Accuracy (Pass@1)** | `evaluate_rule_based.py` | 從模型輸出中擷取最終數字答案（未搭配輸出格式，只計算`####`後的數值），並與標準答案進行比對。|
| **Format Adherence** | `evaluate_dag.py` | 檢查是否符合 Socratic 格式：每步 `Question? ** Response`、計算包在 `<<expr=result>>`、最終答案行以 `####` 開頭、重複問/答扣分。**duplicate_question_ratio**：每題內「被標記為重複問句的步數／該題總步數」；**duplicate_question_ratio_mean**：上述比率在全部測試樣本上的平均。 |
| **BERTScore** | `evaluate_bert.py` | 預測與參考在語意上的 P/R/F1，作為**語意覆蓋率參考**；**無法直接偵測數學推理中的計算錯誤**。 |
| **14B Socratic Quality Score** | `evaluate_with_qwen14b.py` | 以 Qwen2.5-14B 從「拿掉 #### 的推理」算出答案比對；另可打 Socratic 品質 1–5 分。 **［未執行；見 §1 研究範圍與限制］** |

---

## 5. 研究紀錄

以下依訓練與評估輪次簡要紀錄關鍵設定、結果與發現。

### 5.1 第一次訓練

- 訓練時**未設定 system prompt**，推論時若加上 system prompt 會與訓練條件不一致，因此重新訓練。

### 5.2 第二次訓練（本地）

- 訓練完成後與 prompt-only 對照，兩者 accuracy 都不高，但有明顯差異：
  - **Fine-tuned**：accuracy 25.6%，format adherence 95.8%
  - **Prompt-only**：accuracy 6.1%，format adherence 74.0%

**發現：**

1. **Prompt-only 的 format adherence 較低**，可能導致算對卻在 parsing 時無法正確提取答案。後續以 **Qwen2.5-14B** 從推理內容抽取答案（取代純 rule-based），抽樣 20 題發現：14B 容易自行代入計算，因此 14B 抽到的正確率雖較高，但若只看 prompt-only 模型**自身**的答案，正確率不變。**結論**：prompt-only 的低正確率與 parsing 失敗無直接關係。
2. 兩者正確率都不高，可能與 base 模型本身不擅長算數有關；考量本研究重點為「引導思考」，故不更換模型。
3. Fine-tuned 因在整體解題流程上被訓練成較一致，故答對率較高。
4. 在 Apple Silicon（MPS）上，bf16/矩陣運算曾出現 inf/nan，導致 decode 異常，後續選用 colab GPU T4 重新 train 以及 prediction。

### 5.3 第三次訓練（Colab 環境）

- 改在 **Colab（CUDA）** 執行訓練與推論後，accuracy 明顯提升。數值來源：`outputs/20260303/gsm8k_socratic_qwen_eval_*/`。

| 條件 | Reasoning Accuracy | Format Adherence（DAG） | BERTScore (F1) |
|------|--------------------|-------------------------|----------------|
| Fine-tuned | 69.37% | 0.9604 | 0.7723 |
| Prompt-only | 63.84% | 0.7823 | 0.7066 |

**顯著性檢定**  

*n=1319 = GSM8K socratic 測試集全量*（`gsm8k_socratic_alpaca_test.jsonl`，非抽樣）。

| 指標 | 定義／分母 | 檢定方式 | 結果 | 結論 |
|------|------------|----------|------|------|
| **Reasoning Accuracy (A)** | 規則解析 Pass@1（`####` 擷取），n=1319 全量 | Two-proportion z-test | 69.37% vs 63.84%；z ≈ 3.01，p ≈ 0.0026 | 差異達顯著 |
| **Reasoning Accuracy (B)** | 可解析樣本計數（或由推理抽答案），n=1319 | Two-proportion z-test | Fine-tuned 916/1319（69.45%），Prompt-only 946/1319（71.72%）；z = −1.28，p = 0.20 | **未達顯著**（採用為結論） |
| **Format Adherence (DAG)** | 每題 format_score_mean | Paired t-test | Fine-tuned mean 0.9666，Prompt-only mean 0.7284；mean diff = 0.238，t = 21.47，p &lt; 0.001 | 差異達統計顯著 |
| **BERTScore (F1)** | 每題 bertscore_f1 | Paired t-test | Fine-tuned mean 0.7723，Prompt-only mean 0.7066；mean diff = 0.066，t = 34.87，p &lt; 0.001 | 差異達統計顯著 |

**發現：**  
**檢定 A**：若單看「以 `####` 解析出的最終答案」來算正確率（規則解析 Pass@1），Fine-tuned 69.37% vs Prompt-only 63.84%，z ≈ 3.01、p ≈ 0.0026，差異達顯著。**檢定 B**：實際檢視輸出後發現，Prompt-only 格式遵守較差，部分題目因解析失敗而漏提或錯提答案；若依「可成功解析的樣本」重新計數（或改由 14B 從推理內容抽答案），則得到 Fine-tuned 916/1319（69.45%）、Prompt-only 946/1319（71.72%），z = −1.28、p = 0.20，兩者在 accuracy 上差異未達顯著。**結論以檢定 B 為準**：就數值正確率而言，本次無法斷言 fine-tuned 優於 prompt-only；格式遵守度（DAG）與語意覆蓋（BERTScore）的顯著提升則較明確。

---

## 6. 結果分析（對應假說）

| 假說 | 結論與說明 |
|------|------------|
| **H1** | **本研究無法以現有資料集回答此假說。** GSM8K 的 Socratic 格式以拆解問題、導向數值答案為主，與「多輪蘇格拉底式問法」不盡相同；需另外收集對話式引導資料方能驗證 H1。 |
| **H2** | **是**。從 Format Adherence metric 可知，**Fine-tuned 後模型更能遵守制式格式**（如 `Question? ** Response`、`####` 結尾），有利後續 parsing 與自動評估；以同一測試集做 **paired t-test**（每題 format_score_mean）亦可檢定差異顯著性。**侷限**：Fine-tuned 的 duplicate_question_ratio_mean（約 0.13）明顯高於 Prompt-only（約 0.003），表示輸出存在冗餘重複問句，未完全符合「每問一句、一答」的簡潔規範，後續可從訓練資料或 decoding 約束著手改進。 |
| **H3** | **是**。格式一致性與解法可控性顯著較佳；語意覆蓋率（BERTScore）亦顯示一定差距，可依 paired t-test（每題 F1）檢定。**保留說明**：BERTScore 僅反映語意覆蓋率，無法反映數學計算正確性；H3 結論以格式一致性、解法可控性與規則正確率為主，BERTScore 為輔助參考。若受眾為學生，以 fine-tuned 方式可讓解法更**可控**、格式更一致；惟 fine-tuned 答對率約 69%，後續仍有補強空間。 |
---

## 7. 小結與後續

**小結**

- **實務**：在目前資料與設定下，Socratic SFT 能顯著提升**格式遵守度**與語意覆蓋（BERTScore），並使輸出更易解析；數值正確率在本次評估中與 prompt-only 未達顯著差異。**蘇格拉底式問法**則需從資料設計與指標（如 Socratic Score）再加強。
- **環境**：正式評估建議在 **Colab（CUDA）** 或數值穩定的環境執行，以避免 MPS 造成的 accuracy 與格式異常。
- **建議**：首次執行 train、prediction 或 eval 時，建議將 max sample 設為 5，以快速驗證流程與產出，避免投入大量運算資源後才發現寫入等問題；小樣本也便於迅速檢查結果內容。

**後續補強**

- **正確率**：提升 fine-tuned 的答題正確率，可考慮：增加 epoch 或延長訓練、調整 learning rate 或 warmup、擴充或篩選訓練資料／平衡題型、嘗試不同 LoRA rank 或 alpha。
- **重複提問**：`duplicate_question_ratio` Fine-tuned 的**平均值**約 0.13（標準差約 0.20）、Prompt-only 約 0.003（標準差約 0.03），兩者差距大。分佈上，Fine-tuned 約 67% 的題目重複率為 0，**主要由少數高重複率的極端樣本拉高平均值**；後續可優先檢視這些極端題目是否有共同題型特徵，再決定從資料清洗或 decoding 約束（如 `repetition_penalty`）著手改善。