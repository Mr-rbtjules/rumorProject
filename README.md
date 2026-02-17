# Rumors on Social Media Detection Using Machine Learning

An implementation of the CSI (Capture, Score, Integrate) model for classifying social media threads as rumor or non-rumor, combining temporal engagement patterns via LSTM, user credibility scoring via SVD-based embeddings, and text features via sentence transformers with random projection.

---

## Overview

This project tackles rumor detection on social media by reproducing and extending the **CSI framework** (Ruchansky et al., 2017). The model jointly learns from three signals: how engagement unfolds over time (Capture), how credible the participating users are (Score), and what the text says (Integrate). It supports two datasets: **PHEME** (English Twitter) and **Weibo** (Chinese microblog).

---

## Architecture

The model consists of three modules:

### Capture Module (Temporal Patterns)

```
Temporal sequence (seq_len, 122) → Linear(122→100) + LayerNorm + tanh
                                 → LSTM(100→hidden) with packed sequences
                                 → LayerNorm + Linear(hidden→dim_v) + tanh
                                 → Article embedding v_j
```

Each timestep in the sequence represents an hourly bin containing:
- **eta**: number of engagements in the bin
- **delta_t**: time gap since last activity
- **x_u** (20-dim): mean user embedding from SVD
- **x_tau** (100-dim): mean text embedding from sentence transformer + random projection

The LSTM processes variable-length sequences via `pack_padded_sequence` and outputs the final hidden state as the article representation.

### Score Module (User Credibility)

```
User SVD vector y_i (50-dim) → Linear(50→100) + LayerNorm + tanh
                              → Linear(100→1) + sigmoid
                              → Per-user credibility score s_i
```

Scores are averaged per article via `scatter_add_` to produce a single credibility score p_j per thread, scaled by alpha=10 to maintain gradient flow.

### Integrate Module (Classification)

```
concat(v_j, alpha * p_j) → Linear(dim_v+1 → 1) + sigmoid → rumor probability
```

---

## Dimensionality Reduction

### SVD for User Representations

A **user-thread incidence matrix** M (num_users × num_threads) is built where M[u,t] = 1 if user u participated in thread t. Truncated SVD decomposes this matrix:

- **20-dimensional** left singular vectors → user embeddings for the Capture module (x_u)
- **50-dimensional** left singular vectors → user embeddings for the Score module (y_i)

This captures latent patterns of user engagement across threads without end-to-end training.

### Gaussian Random Projection for Text

Tweet text is embedded using SentenceTransformers (384-dim), then reduced to **100 dimensions** via Gaussian Random Projection based on the Johnson-Lindenstrauss lemma, preserving pairwise distances while reducing input dimensionality.

| Dataset | Sentence Transformer |
|---------|---------------------|
| PHEME | `all-MiniLM-L6-v2` |
| Weibo | `paraphrase-multilingual-MiniLM-L12-v2` |

---

## Data Pipeline

1. **Raw parsing**: Traverse PHEME/Weibo directory structure, extract tweets with metadata (user, text, timestamp, thread label)
2. **Text embedding**: SentenceTransformer → 384-dim → Random Projection → 100-dim
3. **User embedding**: Build incidence matrix → Truncated SVD → 20-dim and 50-dim vectors
4. **Temporal binning**: Sort tweets by timestamp, group into 1-hour bins, compute per-bin features (eta, delta_t, mean x_u, mean x_tau)
5. **Standardization**: Log-transform scalar features, Z-score normalize (fit on train only)
6. **Balancing**: Undersample majority class to 50/50, stratified 80/20 train/val split

---

## Training

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | BCE + 0.5 · λ · ‖Wu‖² (λ=0.001) |
| LR scheduler | ReduceLROnPlateau (factor=0.1, patience=8) |
| Gradient clipping | max norm = 5.0 |
| Early stopping | patience = 15 epochs |
| Batch size | 256 |
| Max epochs | 100 |
| Device | MPS / CUDA / CPU (auto-detected) |

A custom collate function handles variable numbers of users per article by flattening user vectors across the batch with index tracking.

---

## Project Structure

```
rumorProject/
├── main.py            # Entry point: device setup, database init, trainer launch
├── model.py           # CSI_model: Capture, Score, Integrate modules
├── trainer.py         # Training loop, evaluation, early stopping, TensorBoard
├── dataBase.py        # PHEME dataset loader and preprocessor
├── dataBaseWeibo.py   # Weibo dataset loader and preprocessor
├── tools.py           # Utilities (timestamp parsing, plotting)
├── config.py          # Paths and random seeds
└── data/
    ├── cache/         # Pickled precomputed tensors
    └── logs/          # TensorBoard logs
```

---

## Getting Started

### Requirements

```
torch
numpy
pandas
scikit-learn
scipy
sentence-transformers
tensorboard
```

### Usage

```python
import rumorProject as RP

# Builds database, computes embeddings, trains model
# Device is auto-detected (MPS/CUDA/CPU)
python main.py
```

Monitor training:
```bash
tensorboard --logdir data/logs/
```

---

## References

- **Natali Ruchansky, Sungyong Seo, Yan Liu** — *CSI: A Hybrid Deep Model for Fake News Detection* ([arXiv:1703.06959](https://arxiv.org/abs/1703.06959))
