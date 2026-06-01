# HARP: Hidden-State-Aware Receiver-Personalized Pruning

Code and resources for **Who Should Listen to Whom? Receiver-Personalized Communication Pruning for Heterogeneous SLM-based Multi-Agent Systems**.

Paper: [OpenReview](https://openreview.net/forum?id=2bxMTM1TM7)

## Why Now: Personal AI PCs Need Smarter Agent Communication

<p>
  <img src="https://img.shields.io/badge/NVIDIA-Personal%20AI%20PC-76B900?logo=nvidia&logoColor=white" alt="NVIDIA Personal AI PC" />
  <a href="https://nvidianews.nvidia.com/news/nvidia-microsoft-windows-pcs-agents-rtx-spark">NVIDIA</a> and <a href="https://blogs.windows.com/windowsexperience/2026/05/31/introducing-a-powerful-new-chapter-for-windows-pcs-accelerated-by-nvidia-rtx-spark/">Microsoft</a> are pushing Windows PCs toward local personal AI agents. HARP addresses the communication layer such systems will need: selective, receiver-personalized, low-latency collaboration among heterogeneous on-device SLM agents.
</p>

Recent personal AI PC announcements make the hardware direction clear: agents are moving closer to the user's device, private data, and interactive workflows. Once multiple small agents can run locally, the next bottleneck is no longer only whether the models fit on the machine. It is **who should listen to whom**: which messages deserve context budget, which sources are useful for each receiver, and how the system avoids redundant, noisy, or adversarial communication.

HARP is built for that missing layer. Instead of assuming that more communication is always better, it uses receiver-side hidden-state evidence to help each heterogeneous SLM agent keep only the incoming sources that are useful for itself under an on-device budget.

> HARP is an independent academic project inspired by public NVIDIA/Microsoft personal AI PC announcements; it is not affiliated with, sponsored by, or endorsed by NVIDIA.

This repository mainly contains the implementation and experimental materials for the **Hidden-State-Aware Receiver-Personalized Pruning (HARP) framework**. The paper introduces the first communication-topology protocol for heterogeneous SLM-MASs and HARP as a practical hidden-state-aware pruning framework built under that protocol.

## What HARP Contributes

### 1. The first communication-topology protocol for heterogeneous SLM-MASs

Heterogeneous small-language-model multi-agent systems (SLM-MASs) are not made of interchangeable agents. Different SLMs have different pretrained knowledge, context capacity, reasoning behavior, and sensitivity to noisy or adversarial messages. HARP formalizes this as a communication-topology problem and organizes the design around four principles:

- **Receiver personalization:** each edge should be evaluated by how useful a source message is to a specific receiver.
- **Complexity adaptiveness:** the topology should adapt to task difficulty, answer divergence, and receiver-side information needs.
- **On-device deployability:** topology construction should be lightweight, local, low-latency, and compatible with deployed SLMs.
- **Adversarial robustness:** communication should reduce the influence of low-quality, noisy, or adversarial senders.

This protocol explains why local agent societies need receiver-specific communication rather than globally dense message passing.

### 2. A hidden-state-aware HARP Score

HARP estimates whether a candidate source message is useful for a particular receiver agent by using the receiver model's hidden states. For each receiver, HARP scores candidate incoming messages and keeps only the most useful sources under a communication budget, producing a sparse and query-specific communication graph.

In short, HARP turns the question:

> Who should listen to whom?

into a receiver-personalized pruning decision based on hidden-state evidence.

### 3. Efficiency and robustness for on-device collaboration

Across the paper experiments, HARP improves or maintains reasoning accuracy while reducing communication overhead. The reported reductions reach up to **17.67%** token usage and **35.68%** latency per question. HARP also improves robustness against adversarial agents, with average relative reductions in attack-induced accuracy drops reaching **91.99%**.

## Repository Scope

This repository is primarily focused on the **HARP framework**, including:

- training a lightweight HARP scorer from receiver-side hidden states;
- running receiver-personalized source selection for heterogeneous SLM agents;
- organizing filtered benchmark samples used in the experiments;
- providing paper figures and supplementary materials.

The communication-topology protocol is described in the paper and reflected in the design of HARP. The main executable code in this repository is the HARP implementation.

## How HARP Works

For a task query and a set of heterogeneous SLM agents, HARP follows this pipeline:

1. Candidate agents generate initial reasoning messages.
2. A receiver agent reads each candidate message.
3. The receiver's hidden states over the message span are extracted.
4. A lightweight HARP scorer predicts the receiver-personalized utility of the candidate edge.
5. For each receiver, HARP keeps the Top-K incoming sources.
6. The selected edges form a sparse receiver-personalized communication topology.

This makes communication dynamic, local, and budget-aware while reducing redundant or harmful information flow.

## Repository Structure

```text
.
|-- assets/figures/                   # Paper figures
|-- data/
|   |-- benchmarks/                   # Filtered benchmark samples
|   `-- motivation/                   # Hidden-state separability diagnostic data
|-- docs/                             # Artifact contracts
|-- harp/                             # HARP scorer modules
|-- scripts/
|   |-- train_harp_scorer.py          # Train the HARP hidden-state scorer
|   |-- run_medical_harp_selection.py # Run HARP receiver-personalized ranking
|   |-- extract_harp_hidden_features.py
|   |-- train_harp_scorer_from_features.py
|   |-- score_harp_edge.py
|   `-- rank_harp_edges.py
|-- supplementary/                    # Supplementary material archive
|-- tests/                            # Minimal scorer checks
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

## HARP Scorer Utilities

HARP scorer utilities expose receiver-conditioned message utility estimation from hidden-state evidence. The score is reported as `P(useful)` and can be used to rank candidate incoming messages under a communication budget.

```text
harp_score: 0.539
score_definition: P(useful)
decision: useful
```

## Run HARP Selection on Medical Reasoning

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
|-- A2/best_harp_s_cnn.pt
|-- A3/best_harp_s_cnn.pt
`-- S/best_harp_s_cnn.pt
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

## Main Experimental Findings

Across the paper experiments, HARP improves or maintains reasoning accuracy while reducing communication overhead. The reported reductions reach up to **17.67%** token usage and **35.68%** latency per question. HARP also improves robustness against adversarial agents, with average relative reductions in attack-induced accuracy drops reaching **91.99%**.

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
