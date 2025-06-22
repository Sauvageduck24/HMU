# Hyper-Memory Unit Transformers: Research Framework

This repository provides a robust framework for benchmarking **standard Transformers** and **Hyper-Memory Unit Transformers (HTransformers)** in tasks involving long-term memory and imagination. The codebase is designed to support reproducible research, as presented in our paper:

- [Paper (English, PDF)](src/data/paper/HMU_en.pdf)
- [Artículo (Español, PDF)](src/data/paper/HMU_es.pdf)

## Overview

The framework enables:
- Direct comparison between standard Transformer architectures and HTransformer variants.
- Experiments on synthetic and real-world NLP datasets (GLUE, AGNews, CommonGen, ROCStories, SST-2, etc.).
- Evaluation on memory retention, imagination, and standard classification tasks.
- Generation of publication-ready metrics and visualizations.

## Model Architectures

- **Standard Transformer**: Implements the canonical encoder-decoder architecture for sequence modeling.
- **HTransformer (Hyper-Memory Unit Transformer)**: Augments the encoder with a Hyper-Memory Unit (HMU) that compresses and fuses latent representations, enhancing long-term memory and imagination capabilities.

Both models are implemented in PyTorch and are configurable via command-line arguments.

## Installation

1. **Clone the repository**
   ```bash
   git clone <repo_url>
   cd <repo_dir>
   ```
2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running Experiments

The main entry point is `run_experiments.py`. All experiments, training, evaluation, and benchmarking are managed via this script.

### Basic Usage

```bash
python run_experiments.py [OPTIONS]
```

### Key Arguments

| Argument           | Description                                                                                 | Default         |
|--------------------|---------------------------------------------------------------------------------------------|-----------------|
| `--mode`           | Experiment mode: `all` (train+eval), `train` (only train), `benchmark` (only eval)          | all             |
| `--model`          | Model type: `transformer`, `htransformer`, `htransformerafterdecoder`                       | htransformer    |
| `--dataset`        | Dataset: `synthetic`, `commongen`, `rocstories`, `agnews`, `sst2`, `mrpc`, `rte`, `cola`, `qnli`, `mnli`, `glue` | synthetic |
| `--epochs`         | Number of training epochs                                                                   | 1               |
| `--batch_size`     | Training batch size                                                                         | 32              |
| `--d_model`        | Model hidden size (dimension)                                                               | 256             |
| `--seq_len`        | Sequence length (for synthetic data)                                                        | 128             |
| `--dataset_size`   | Number of samples (for synthetic data)                                                      | 20000           |
| `--grad_accum`     | Gradient accumulation steps                                                                 | 1               |
| `--max_length`     | Max sequence length for tokenization                                                        | 32              |
| `--max_samples`    | Max samples per dataset (for large datasets)                                                | None            |
| `--proof`          | Enable proof mode (quick tests with tiny datasets)                                          | False           |
| `--eval_batch_size`| Batch size for evaluation                                                                   | 8               |
| `--tokenizer_name` | HuggingFace tokenizer/model name (e.g., `t5-small`)                                        | t5-small        |

### Example Commands

**Train and evaluate on SST-2 (GLUE):**
```bash
python run_experiments.py --dataset sst2 --mode all --epochs 5 --batch_size 16
```

**Quick proof mode (for debugging):**
```bash
python run_experiments.py --dataset mrpc --mode all --proof
```

**Benchmark a trained model:**
```bash
python run_experiments.py --dataset agnews --model htransformer --mode benchmark
```

**Run all GLUE tasks:**
```bash
python run_experiments.py --dataset glue --mode all --epochs 3 --batch_size 16
```

### Output and Results
- Trained model checkpoints are saved in the `results/` directory (e.g., `htransformer_sst2.pt`).
- Comparative metrics and summaries are exported as CSV files (e.g., `comparative_summary.csv`, `glue_comprehensive_results.csv`).
- Training curves and evaluation plots are generated as PNG images for publication use.

## Supported Datasets
- **Synthetic**: Custom patterns for memory and imagination benchmarking.
- **GLUE**: MRPC, RTE, CoLA, QNLI, SST-2, MNLI (see [README_GLUE.md](README_GLUE.md) for details).
- **AGNews**: News topic classification.
- **CommonGen**: Generative commonsense reasoning.
- **ROCStories**: Story completion and narrative understanding.
- **SST-2**: Sentiment analysis.

## Evaluation Metrics
Implemented in `src/evaluation/metrics.py`:
- **Memory Accuracy**: Sequence-level and exact match accuracy.
- **Memory Retention**: Cosine similarity of representations over time.
- **Imagination Quality**: BLEU, ROUGE-2, sequence coherence, diversity, perplexity.
- **Classification Metrics**: Accuracy, precision, recall, F1 (for GLUE/AGNews).

## Visualization
- Training and evaluation curves are saved in `results/` (e.g., `training_curves_comparison.png`).
- Memory and imagination metrics are visualized for direct model comparison.

## Reproducibility & Experiment Tracking
- All experiments are tracked with [Weights & Biases (wandb)](https://wandb.ai/).
- Random seeds are set for full reproducibility.

## Citing This Work
If you use this codebase or results in your research, please cite the paper using the provided PDFs:
- [English PDF](src/data/paper/HMU_en.pdf)
- [PDF en Español](src/data/paper/HMU_es.pdf)

---

For questions, open an issue or contact the authors via the paper. 