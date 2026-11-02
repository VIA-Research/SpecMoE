# SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding

SpecMoE is an inference system for Mixture-of-Experts (MoE) models that combines
expert offloading with self-assisted speculative decoding. It uses a subset
of the model's experts as a draft model, reducing CPU–GPU data transfers and
accelerating inference without training a separate draft model.

For more details, see our [paper](https://dl.acm.org/doi/pdf/10.1145/3770743.3803936)
published at DAC 2026 and the [longer version](https://arxiv.org/abs/2604.10152).

## Table of Contents

- [Setup](#setup)
- [Project Structure](#project-structure)
- [Example](#example)
- [Reproducing Results](#reproducing-results)
- [Acknowledgment](#acknowledgment)
- [Citation](#citation)

## Setup

```bash
git clone https://github.com/bangjh0430/SpecDecOffload.git
cd SpecDecOffload
conda create -n specmoe python=3.9.23
conda activate specmoe
bash setup.sh
```

Download the [NLLB-MoE 54B checkpoint](https://huggingface.co/facebook/nllb-moe-54b),
including tokenizer files, to `$MODEL_DIR/nllb-moe-54b/`.

```bash
export MODEL_DIR=/path/to/model_checkpoints
export OFFLOAD_DIR=/path/to/ssd/scratch  # only for disk offloading
```

## Project Structure

```text
setup.sh           # Environment setup
requirements.txt   # Dependencies
run.py             # Single experiment
run.sh             # Experiment configurations
configs/           # Experiment settings
specdec/           # Speculative decoding
modelings/         # NLLB extensions
patches/           # Runtime hooks
config.py          # Model and dataset settings
device_map.py      # Model placement
affinity_tables/   # Routing inputs
hot_experts/       # Expert presets
scripts/           # Experiment and verification scripts
tests/             # Tests
```

## Example

```bash
# Greedy
python run.py --method specdec --lang fr --batch-size 128 --gamma 10
python run.py --method cache   --lang fr --batch-size 128
python run.py --method nocache --lang fr --batch-size 128

# Sampling
python run.py --method specdec --decoding sampling --temperature 0.8 --top-k 50 --top-p 0.95
```

### CLI reference

```bash
python run.py --help
```

| Argument | Default |
| --- | --- |
| `--method {specdec,cache,nocache}` | required |
| `--target` | `nllb-moe-54b` |
| `--dataset` | `wmt` |
| `--lang {fr,de}` | `fr` |
| `--offload-device {cpu,disk}` | `cpu` |
| `-b`, `--batch-size` | `128` |
| `-n`, `--input-len` | `128` |
| `--output-len` | `32` |
| `-g`, `--gamma` | `10` |
| `-D`, `--num-fixed-experts` | `4` |
| `--num-cached-experts` | `13` |
| `--decoding {greedy,sampling}` | `greedy` |
| `-T`, `--temperature` | `1` for sampling |
| `--top-k` | `0` |
| `--top-p` | `1` |
| `--seed` | `240725` |
| `--num-samples` | `batch_size * 10` |
| `-o`, `--output` | generated filename |
| `--measure {movement,overlap}` | off |
| `-v`, `--verbose` | off |
| `-q`, `--quiet` | off |

## Reproducing Results

`run.sh` uses [configs/nllb.json](configs/nllb.json). Override settings with CLI
options or `--config <path>`. Figure and table numbers below refer to the
[longer version](https://arxiv.org/abs/2604.10152).

```bash
# Fig. 10–11: nocache, cache, specdec
bash run.sh

# Fig. 12
bash run.sh --measure movement

# Table III
bash run.sh --method specdec --batch-sizes 256 --num-fixed-experts 2 4 8 16

# Fig. 13: Affinity curve
bash run.sh --method specdec --batch-sizes 128 --gamma 10 12 15 17 20

# Fig. 16: requires OFFLOAD_DIR and separate host page-cache control
bash run.sh --offload-device disk --input-len 64 --output-len 32
```

## Acknowledgment

This research is funded by the generous support from the following organizations:

- Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.RS-2025-02264029, Implementation and Validation of an AI Semiconductor-Based Data Center Composable Cluster Infrastructure, 30\%),
(No.RS-2024-00438851, (SW Starlab) High-performance Privacy-preserving Machine Learning System and System Software),
(No.RS-2024-00402898, Simulation-based High-speed/High-Accuracy Data Center Workload/System Analysis Platform),
(No. RS-2024-00457882, AI Research Hub Project)
- Samsung Research Funding Center of Samsung Electronics (SRFC-IT2402-03)
- Samsung Advanced Institute of Technology, Samsung Electronics Co., Ltd.

## Citation

```bibtex
@inproceedings{bang2026specmoe,
  title={SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding},
  author={Bang, Jehyeon and Cho, Eunyeong and Hwang, Ranggi and Chung, Jinha and Rhu, Minsoo},
  booktitle={2026 63rd ACM/IEEE Design Automation Conference (DAC)},
  year={2026},
  doi={10.1145/3770743.3803936}
}
```
