# NeurIPS2026-HARP Framework

Official implementation package for **HARP: Hidden-State-Aware Receiver-Personalized Pruning**, a communication-pruning framework for heterogeneous small-language-model multi-agent systems (SLM-MAS).

HARP asks a practical question in collaborative SLM reasoning: **who should each receiver agent listen to?** Instead of using dense communication or a single global topology, HARP estimates the receiver-specific utility of candidate incoming messages from the receiver model's hidden states, then keeps only the most useful sources under a communication budget.

## Overview

Heterogeneous SLM-based multi-agent systems can improve on-device reasoning by combining agents with different pretrained knowledge, context capacity, and reasoning behavior. However, dense communication is often inefficient and can even hurt performance because weaker receivers may be sensitive to redundant, noisy, or adversarial messages.

HARP introduces a receiver-personalized pruning pipeline:

1. Generate candidate reasoning messages from heterogeneous SLM agents.
2. Feed each candidate message into the target receiver model.
3. Extract receiver-side hidden states over the candidate message span.
4. Score the candidate edge with a lightweight HARP scorer.
5. Keep the Top-K incoming sources for each receiver to form a sparse, query-specific communication graph.

The resulting topology is dynamic, sparse, receiver-personalized, and compatible with local SLM deployment.

## Paper Highlights

- Proposes a heterogeneous SLM-MAS communication topology protocol built around receiver personalization, complexity adaptiveness, on-device deployability, and adversarial robustness.
- Introduces HARP Score, a hidden-state-aware utility surrogate for estimating whether a source message is useful to a receiver.
- Evaluates HARP on four medical reasoning benchmarks: MedQA, MedMCQA, PubMedQA, and MMLU-Medical.
- Tests generalization on five broader reasoning benchmarks: MMLU, GSM8K, MultiArith, SVAMP, and AQuA.
- Reports improved or maintained accuracy while reducing token usage and latency; in the paper experiments, reductions reach up to 17.67% token usage and 35.68% latency per question.
- Shows strong robustness against adversarial agents, with average relative reductions in attack-induced accuracy drops up to 91.99%.

## Repository Structure

```text
.
|-- assets/figures/                   # Paper figures
|-- data/
|   |-- benchmarks/                   # Filtered benchmark samples used for evaluation
|   `-- motivation/                   # Diagnostic data for HARP hidden-state separability
|-- scripts/
|   |-- train_harp_scorer.py          # Train the hidden-state HARP scorer
|   `-- run_medical_harp_selection.py # Run receiver-personalized HARP ranking on medical tasks
|-- supplementary/                    # Supplementary material archive
|-- requirements.txt
`-- README.md
```

## Installation

```bash
git clone https://github.com/Horizon-AILab/NeurIPS2026-HARP-Framework.git
cd NeurIPS2026-HARP-Framework
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train a HARP Scorer

The scorer is trained from positive and negative receiver-personalized message examples. It extracts hidden states from a frozen local SLM backbone and trains a lightweight CNN classifier.

```bash
python scripts/train_harp_scorer.py \
  --model_path /path/to/local/receiver-model \
  --pos_train_path /path/to/positive_train.json \
  --neg_dataset1_path /path/to/negative_train_1.json \
  --neg_dataset2_path /path/to/negative_train_2.json \
  --local_data_path outputs/train_cache.json \
  --output_dir outputs/harp_scorer/A3 \
  --device_llm cuda:1 \
  --device_cnn cuda:0
```

The training script supports selecting all hidden layers, the last layer, or front/middle/back layer groups through `--use_layers` and `--use_part`.

## Run Medical HARP Selection

The medical ranking script expects:

- filtered medical benchmark files under `data/benchmarks/`;
- cached local agent outputs, configured by `HARP_AGENT_OUTPUTS_JSONL`;
- local receiver model checkpoints under `models/`;
- trained HARP scorer checkpoints under `checkpoints/harp_score_modules/`.

Default model paths follow this layout:

```text
models/
|-- Llama-3.1-8B-Instruct/
|-- Meta-Llama-3-8B-Instruct/
`-- Qwen3-8B/

checkpoints/harp_score_modules/
|-- A2/best_deeptextcnn.pt
|-- A3/best_deeptextcnn.pt
`-- S/best_deeptextcnn.pt
```

Run:

```bash
export HARP_AGENT_OUTPUTS_JSONL=/path/to/precomputed_medical_agent_outputs.jsonl
python scripts/run_medical_harp_selection.py
```

Useful environment variables:

```bash
export HARP_DATA_DIR=/path/to/benchmark_jsonl_dir
export HARP_OUTPUT_ROOT=/path/to/output_dir
export HARP_PROJECT_ROOT=/path/to/project_root
export HARP_SCORE_ROOT=/path/to/harp_score_modules
```

The script writes detailed per-question outputs, summary tables, HARP checkpoint metadata, and Excel reports to `outputs/medical_harp_selection/` by default.

## Benchmarks

The included filtered samples cover:

- Medical reasoning: MedQA, MedMCQA, PubMedQA, MMLU-Medical.
- General reasoning: MMLU, GSM8K, MultiArith, SVAMP, AQuA.

These files are intended to make the evaluation protocol transparent. For full benchmark usage, please follow the licenses and terms of the original datasets.

## Citation

```bibtex
@misc{harp2026,
  title  = {Who Should Listen to Whom? Receiver-Personalized Communication Pruning for Heterogeneous SLM-based Multi-Agent Systems},
  author = {Anonymous},
  year   = {2026},
  note   = {NeurIPS 2026 submission}
}
```

## License

The license will be added by the authors.
