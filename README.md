# Fine-tuning LLM for GSM8K Socratic

Fine-tune Qwen2.5-3B-Instruct on the GSM8K Socratic subset using LLaMA-Factory, with multi-metric evaluation for Socratic-style math tutoring responses.

## Features

- **Socratic-format SFT**: Train on the `openai/gsm8k` socratic subset with a research-oriented instruction (Question? ** Answer with `<<expr=result>>`, final answer after `####`)
- **Multi-metric evaluation**:
  - Rule-based: Answer Accuracy (Pass@1 from `####`/`###` extraction)
  - DAG: Structural format comparison (steps, `<<expr=result>>`, duplicate detection)
  - BERTScore: Token-level semantic similarity (precision/recall/F1)
  - LLM grading: Qwen2.5-14B for Reasoning (from stripped reasoning) and Socratic Score (1–5) *(optional; see note below)*
- **Statistical significance**: Script for two-proportion z-test (accuracy) and paired t-tests (DAG, BERTScore) between fine-tuned vs prompt-only.

## Note on GSM8K Socratic Dataset

GSM8K was designed to train models to *answer* math problems; its Socratic format therefore emphasizes **拆解問題** (problem decomposition) — breaking down the task into steps toward a numerical answer. This project initially aimed for **引導問題** (guiding questions) — a more exploratory, tutor-like Socratic style. Because the fine-tuned outputs did not meet the ideal Socratic quality in this sense, LLM grading–related metrics were not pursued further. The evaluation pipeline still supports rule-based, DAG, BERTScore, and optional LLM grading for reference.

## Prerequisites

- Python 3.10+
- LLaMA-Factory (via git submodule)
- HuggingFace datasets
- For evaluation: `bert-score`, `transformers`, `torch`, `bitsandbytes` (4-bit models)
- Optional (Apple Silicon): `mlx`, `mlx-lm` for MLX backend

## Installation

```bash
# Clone repo with submodule
git clone --recursive https://github.com/YOUR_ORG/Fine-tunning-LLM.git
cd Fine-tunning-LLM

# Create venv and install deps
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e LLaMA-Factory  # or follow LLaMA-Factory install docs

# Apple Silicon (optional): pip install mlx mlx-lm
```

## Quick Start

Run from the project root.

1. **Prepare data** (downloads GSM8K socratic, converts to Alpaca format):

   ```bash
   python scripts/prepare_socratic_dataset.py
   ```

2. **Fine-tune**:

   ```bash
   llamafactory-cli train socratic_train_config.yaml
   ```

   Output: e.g. `outputs/gsm8k_socratic_qwen/` (or path set in config).

3. **Run evaluation predictions**:

   ```bash
   llamafactory-cli train socratic_promptonly_eval.yaml   # base model
   llamafactory-cli train socratic_finetuned_eval.yaml    # fine-tuned adapter
   ```

4. **Evaluate**:

   ```bash
   python evaluate_rule_based.py <eval_dir_promptonly>
   python evaluate_rule_based.py <eval_dir_finetuned>

   python evaluate_dag.py <eval_dir_promptonly>
   python evaluate_dag.py <eval_dir_finetuned>

   python evaluate_bert.py <eval_dir_promptonly>
   python evaluate_bert.py <eval_dir_finetuned>
   ```

   Optional LLM grading:

   ```bash
   python evaluate_with_qwen14b.py <eval_dir> --model Qwen/Qwen2.5-14B-Instruct --quantize 4bit
   ```

5. **Statistical significance** (fine-tuned vs prompt-only):

   ```bash
   python scripts/significance_accuracy.py --dirs <finetuned_eval_dir> <promptonly_eval_dir>
   ```

## Project Structure

```
Fine-tunning-LLM/
├── README.md
├── requirements.txt
├── socratic_train_config.yaml      # SFT config
├── socratic_promptonly_eval.yaml   # Eval: base model
├── socratic_finetuned_eval.yaml    # Eval: fine-tuned adapter
├── chat_qwen25_3b.yaml             # Chat: base Qwen2.5-3B
├── chat_fine-tuned-qwen25_3b.yaml  # Chat: fine-tuned adapter
├── evaluate_rule_based.py          # Rule-based answer accuracy
├── evaluate_bert.py                # BERTScore
├── evaluate_dag.py                 # DAG structure / format metrics
├── evaluate_with_qwen14b.py        # LLM-based grading (optional)
├── scripts/
│   ├── prepare_socratic_dataset.py # Data preparation
│   ├── get_top_socratic_samples.py # Top samples by Socratic Score
│   ├── print_dag_issues.py         # Duplicate answers & DAG violations report
│   ├── significance_accuracy.py    # Significance tests (z-test, t-test)
│   └── run_vllm_predict.py         # vLLM-based prediction
├── data/custom/                    # Generated dataset (from prepare script)
├── outputs/                        # Trained models, predictions, eval results
├── notebook/
│   └── run_on_collab.ipynb         # Colab workflow
├── docs/
│   └── Research Design and Evaluation Records.md  # 研究設計與評估紀錄
└── LLaMA-Factory/                  # Submodule
```

## Evaluation Metrics

| Metric                    | Script                    | Description                                        |
|---------------------------|---------------------------|----------------------------------------------------|
| Reasoning (Accuracy)      | `evaluate_rule_based.py`  | Pass@1 from `####`/`###` answer extraction         |
| Format (DAG)              | `evaluate_dag.py`         | Step/format/structural scores, duplicate detection |
| BERTScore (P/R/F1)        | `evaluate_bert.py`       | Token-level semantic similarity                    |
| Reasoning (14B)           | `evaluate_with_qwen14b.py`| 14B infers answer from stripped reasoning          |
| Socratic Score (1–5)      | `evaluate_with_qwen14b.py`| 14B rates Socratic style                           |
| Significance              | `scripts/significance_accuracy.py` | z-test (accuracy), paired t-test (DAG, BERT)  |

*LLM grading (Reasoning 14B, Socratic Score) was not fully pursued due to the dataset philosophy mismatch described above.*

## Documentation

- [Research Design and Evaluation Records](docs/Research%20Design%20and%20Evaluation%20Records.md) — 研究設計與評估紀錄、假說、實驗參數與結果摘要

## Colab

Use [notebook/run_on_collab.ipynb](notebook/run_on_collab.ipynb) for a Google Colab workflow. Set `output_dir` in `socratic_train_config.yaml` to `/content/drive/MyDrive/llm_outputs/...` if saving checkpoints to Google Drive.
