# MSP-DA

This repository contains the code for:

**Unsupervised Domain Adaptation for Text Classification via Meta Self-Paced Learning** \
*Nghia Trung Ngo, Linh Ngo Van, Thien Huu Nguyen* \
ACL-IJCNLP 2021

## Method

MSP-DA addresses unsupervised domain adaptation (UDA) for text classification by integrating meta-learning with a neural self-paced learning (SPL) module. The key idea is to dynamically partition labeled source data into a **meta-source** (easy, low-loss) set and a **meta-target** (hard, high-loss) set, simulating a virtual domain shift. A MAML-style inner loop trains on the easy meta-source subset, and the outer loop evaluates on the hard meta-target subset—this signal updates the SPL module's age threshold and the layer-wise DANN weighting network simultaneously.

**Architecture overview:**
- Fixed BERT-base encoder augmented with adapter layers (bottleneck size 96) after every feed-forward sub-block
- Per-layer DANN heads (12 total) taking adapter bottleneck representations → Gradient Reversal Layer → binary domain classifier
- Instance-wise weighting head (SPL_CE): maps per-sample CE losses through a small MLP with learnable age parameter λ_a
- Layer-wise weighting head (MWN_DA): aggregates adapter bottleneck representations → softmax → 12 DANN layer weights
- Optional pseudo-label self-training on easy target examples

## Installation

```bash
git clone https://github.com/user/msp-da.git
cd msp-da
pip install -r requirements.txt
pip install -e .
```

**Requirements:** Python 3.7+, PyTorch ≥ 1.9, Transformers ≥ 4.10

## Data Preparation

### ACE-05 (Event Detection)

Requires an LDC license. After obtaining raw data, preprocess with:
```bash
python scripts/preprocess_ace05.py \
    --input_dir /path/to/raw/ace2005 \
    --output_dir data/ace2005/
```

### FDU-MTL (Sentiment Analysis)

Publicly available:
```bash
git clone https://github.com/FrankWork/fudan_mtl_reviews
cp -r fudan_mtl_reviews/data/ data/fdu-mtl/
```

See `data/README.md` for full details.

## Usage

### Training

```bash
# ACE-05 Event Detection (source: bn+nw, target: bc / cts / wl)
python train.py --config configs/ace05_bc.yaml
python train.py --config configs/ace05_cts.yaml
python train.py --config configs/ace05_wl.yaml

# Run all ACE-05 settings
bash scripts/train_ace05.sh all

# FDU-MTL Sentiment Analysis (target: MR)
python train.py --config configs/fdu_mtl_MR.yaml
```

To skip source pre-training and load a checkpoint:
```bash
python train.py --config configs/ace05_bc.yaml \
    --skip_pretrain \
    --pretrain_ckpt checkpoints/ace05_bc/pretrain.pt
```

Key hyperparameters in `configs/ace05_bc.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-5 | Outer optimizer LR (adapter + heads) |
| `meta_train_lr` | 1e-4 | Inner step LR (MAML adaptation) |
| `meta_val_lr` | 1e-3 | Meta-val optimizer LR (SPL + MWN) |
| `batch_size` | 150 | Mini-batch size |
| `target_ratio` | 0.2 | Fraction of target samples per batch |
| `num_meta_train_steps` | 2 | Number of inner gradient steps |
| `meta_test_beta` | 1.0 | Weight of meta-val loss in total objective |
| `pseudo_label` | true | Use pseudo-labels for easy target samples |
| `dann_wtype` | `mwn` | DANN layer weighting: `mwn`, `uniform`, `anneal_up`, `anneal_dw` |

### Evaluation

```bash
python evaluate.py \
    --config configs/ace05_bc.yaml \
    --checkpoint checkpoints/ace05_bc/best_model.pt
```

## Results

### Event Detection on ACE-05 (F1, source: bn+nw)

| System | In-domain | → bc | → cts | → wl | Avg OOD |
|--------|:---------:|:----:|:-----:|:----:|:-------:|
| BERT | 74.1 | 71.1 | 71.5 | 56.4 | 66.3 |
| BERT+DANN | 74.7 | 71.5 | 62.5 | 56.3 | 63.4 |
| Focal Loss | 77.9 | 72.2 | 70.1 | 59.0 | 67.1 |
| **MSP-DA (Ours)** | **77.7** | **75.8** | **76.1** | **64.8** | **72.2** |

### Sentiment Analysis on FDU-MTL (Average Accuracy)

| System | Avg Acc |
|--------|:-------:|
| ASP-MTL | 90.1 |
| BERT | 91.5 |
| BertMasker | 91.5 |
| **MSP-DA (Ours)** | **93.0** |

### Ablation Study (ACE-05, F1)

| Variant | → bc | → wl |
|---------|:----:|:----:|
| MSP-DA − mSPL | 74.6 | 57.4 |
| MSP-DA − DANN | 74.2 | 56.3 |
| MSP-DA − PL | 74.3 | 57.0 |
| **MSP-DA (full)** | **75.8** | **64.8** |

## Citation

```bibtex
@inproceedings{ngo-etal-2021-unsupervised,
    title={Unsupervised Domain Adaptation for Text Classification via Meta Self-Paced Learning},
    author={Ngo, Nghia Trung and Van, Linh Ngo and Nguyen, Thien Huu},
    booktitle={Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
    year={2021},
}
```

## Acknowledgements

This work builds on [adapter-transformers](https://github.com/adapter-hub/adapter-transformers) and [HuggingFace Transformers](https://github.com/huggingface/transformers).

## License

MIT
