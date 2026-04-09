# A Readability-Driven Curriculum Learning Method for Data-Efficient Small Language Model Pretraining

[![Paper](https://img.shields.io/badge/Paper-MDPI_Mathematics-blue.svg)](https://www.mdpi.com/2227-7390/13/20/3300)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)]()

This repository contains the official PyTorch implementation of the pretraining pipeline for the paper **"A Readability-Driven Curriculum Learning Method for Data-Efficient Small Language Model Pretraining"** (Published in *Mathematics*, MDPI).

## 📖 Overview

Large language models (LLMs) demand substantial computational and data resources, highlighting the need for efficient training approaches for small language models (SLMs). While Curriculum Learning (CL) based on linguistic difficulty has been explored, previous methods relying on complex linguistic indices are often computationally expensive and difficult to interpret.

Inspired by the cognitive and linguistic efficiency observed in human language acquisition, we propose a **readability-driven curriculum learning method based on the Flesch Reading Ease (FRE) score**. This provides a simple, interpretable, and cognitively motivated measure of text difficulty. Our method yields consistent improvements over baseline models without curriculum learning, achieving substantial gains on BLIMP and MNLI benchmarks.

This repository provides the pipeline to replicate the **Curriculum Pretraining** phase of our proposed methodology.

## 🧠 Methodology: How it Works

To fully replicate the paper's methodology, the process is divided into two main steps: Data Rearrangement (Dataset Preparation) and Curriculum Pretraining (This Repository).

### 1. Data Rearrangement (Dataset Preparation)

![Data Rearrangement Method](assets/figure_final.png)
*(Figure 1: Overview of the Readability-Driven Data Rearrangement Process)*

As illustrated in the figure above, the core of our approach lies in rearranging the training data based on readability. Before running the training pipeline, the original corpus must be processed:
1. **Granularity Splitting:** The original dataset is divided into specific structural units: Sentences, Groups (small chunks of sentences), or Paragraphs.
2. **FRE Scoring:** Each unit is evaluated using the Flesch Reading Ease (FRE) formula.
3. **Difficulty Sorting:** Based on the FRE scores, the units are sorted and grouped into distinct difficulty levels: **Level 1 (Easy, High FRE)**, **Level 2 (Medium)**, and **Level 3 (Hard, Low FRE)**. 

*Note: You need to prepare your dataset in this sorted format (e.g., `phase_1.json`, `phase_2.json`, `phase_3.json`) prior to using this repository.*

### 2. Curriculum Pretraining (This Repository)

Once your dataset is arranged by difficulty, this repository handles the curriculum pretraining pipeline. It introduces two crucial, cognition-inspired features:

* **Sentence-Boundary Preserving Chunking:** When tokenizing and chunking the data to fit the model's maximum context length (e.g., 1024 for GPT-2, 512 for BERT), standard methods often ruthlessly cut sentences in half. Our `dataset.py` logic strictly chunks data based on sentence boundaries (`.`, `!`, `?`), ensuring the model learns from complete, semantically intact sentences—just as humans do.
* **Cognitive-Inspired Evaluation (Word-count Checkpointing):** Human language acquisition is typically measured by the number of words exposed to the learner, not arbitrary "epochs" or "steps." Our custom callback in `utils.py` precisely tracks the **cumulative number of real words** the model has trained on (excluding special tokens like `[PAD]`, `[CLS]`, `[SEP]`) and saves checkpoints at meaningful milestones (e.g., 1M, 10M, 100M words).

## 📂 Repository Structure

```text
.
├── configs/
│   ├── gpt_config.yaml      # Configuration for GPT-2 training
│   └── bert_config.yaml     # Configuration for BERT training
├── scripts/
│   └── train.sh             # Shell script to launch training
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Sentence-boundary preserving chunking & tokenization
│   ├── model.py             # GPT-2 / BERT model initialization 
│   └── utils.py             # Word-count tracking and checkpointing callback
├── main.py                  # Main entry point for curriculum training loop
└── requirements.txt         # Required Python packages


## 🚀 Quick Start

### 1. Requirements
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Following the methodology described above, prepare your JSON datasets sorted by FRE scores. Your files should be formatted with a `text` key: `{"text": "..."}`.

### 3. Configuration
Open `configs/gpt_config.yaml` or `configs/bert_config.yaml` and update the `[TODO]` sections to match your local paths:
```yaml
# Example from configs/gpt_config.yaml
phases:
  easy: "./data/phase_1_easy.json"
  medium: "./data/phase_2_medium.json"
  hard: "./data/phase_3_hard.json"
```

### 4. Run Training
Use the provided shell script to start the training. The pipeline will automatically load the Easy data, train the model, save word-count checkpoints, and then sequentially move on to the Medium and Hard phases.

**To train GPT-2 (Causal LM):**
```bash
bash scripts/train.sh gpt2
```

**To train BERT (Masked LM):**
```bash
bash scripts/train.sh bert
```

## 📝 Citation
If you find this repository or our methodology useful for your research, please cite our paper:

```bibtex
@article{kim2025readability,
  title={A Readability-Driven Curriculum Learning Method for Data-Efficient Small Language Model Pretraining},
  author={Kim, Suyun and Park, Jungwon and Kim, Juae},
  journal={Mathematics},
  volume={13},
  number={20},
  pages={3300},
  year={2025},
  publisher={MDPI},
  url={[https://www.mdpi.com/2227-7390/13/20/3300](https://www.mdpi.com/2227-7390/13/20/3300)}
}
```

## 📜 License
* **Code License:** The code in this repository is licensed under the [MIT License](LICENSE).
* **Paper License:** The published article is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
