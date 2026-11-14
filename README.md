# Team Synora — ML Gradient Wars 2026
## Official Submission & Reproduction Repository: UPI Fraud Rings Challenge

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Competition](https://img.shields.io/badge/Competition-ML%20Gradient%20Wars%202026-brightgreen)](https://hack4hack.com)
[![Team](https://img.shields.io/badge/Team-Synora-orange.svg)](#team)

---

## 📌 Executive Summary

This repository contains the complete, production-grade codebase, end-to-end multi-stage stacking pipeline, and reproduction materials developed by **Team Synora (`atharva-awade`)** for the **ML Gradient Wars 2026: UPI Fraud Rings 48-Hour Live Round** (13 Nov 2026, 10:00 AM IST – 15 Nov 2026, 10:00 AM IST).

### The Challenge
Predict **12 fine-grained transaction classes** (`legit` + 11 distinct fraud ring typologies) across **260,582 October transactions**, evaluated strictly under unweighted **Macro-F1**:

$$\text{Macro-F1} = \frac{1}{12}\sum_{c=1}^{12} F_{1, c}$$

### The 11 Fraud Typologies:
1. `mule_account`
2. `velocity_bot`
3. `layering_chain`
4. `sim_swap`
5. `account_takeover`
6. `phishing_collect`
7. `qr_swap`
8. `refund_abuse`
9. `fake_cashback`
10. `loan_app_extortion`
11. `merchant_collusion`

---

## 🏆 Key Breakthroughs & Architectural Innovations

### 1. The 50% Test Density Scaling Trap
- **The Issue:** Training data spans July, August, and September (~520,000 transactions/month), while the October test set contains exactly 260,582 rows (a 50% sample). Unscaled rolling counts (e.g., transactions in last 1 hour, burst intensity) were halved in test data, preventing tree splits calibrated on full training density from firing.
- **The Fix:** Applied a calibrated $2.0\times$ density scale to all 65 count-family features on test data to ensure statistical invariance across splits.

### 2. Transductive Multi-Graph & Temporal Account Ledger
- **Bidirectional Ledger:** Extracted 23 pass-through features including inbound-to-outbound pass-through latency ($\Delta t_{\text{pass}} \le 10\text{m}$), retention ratio $R \in [0.85, 1.05]$, and account conduit indices.
- **Transductive Graph Engine:** Built a unified bipartite transaction graph over $V_{\text{train}} \cup V_{\text{test}}$, extracting PageRank, directed 3-cycle wash loops, and Strongly Connected Components (SCC).
- **Spectral Embeddings:** 10-dimensional TruncatedSVD bipartite account $\times$ merchant homophily vectors.

### 3. Orthogonal Fraud Detection Fusion
- Discovered that structural network rings (detected via graph topology and ledger balance flows) and temporal behavioral bursts (detected via micro-second balance jumps and emulator device multiplexing) operate in nearly disjoint subspaces (only 54 rows overlap). Fusing these orthogonal feature spaces drove recall from 39% to over 78% while maintaining $\ge 68\%$ precision.

### 4. 3-Stage GPU Stacking & Bagged Ensembles
- **Stage 1:** Base feature extraction and 5-fold stratified cross-validation across LightGBM, CatBoost, and XGBoost models.
- **Stage 2:** Multi-hop spatial and neighbor probability stacking across entity graphs.
- **Stage 3:** Bagged probability ensembling and holdout-calibrated probability alignment (`build_aligned_submission.py`).

---

## 📁 Repository Structure

```text
├── src/                                   # Core modular feature engineering & training library
│   ├── __init__.py
│   ├── config.py                          # Global configuration, paths, seeds, and label maps
│   ├── cache.py                           # Parquet-based caching layer for high-throughput I/O
│   ├── graph_engine.py                    # Transductive bipartite transaction graph features
│   ├── ledger.py                          # Bidirectional temporal account ledger engine
│   ├── typology_features.py               # Domain-specific features for 11 fraud typologies
│   ├── spectral_embeddings.py             # TruncatedSVD bipartite graph spectral embeddings
│   ├── target_encoding.py                 # Out-of-fold Bayesian target encoding for entities
│   ├── quota_calibration.py              # Prior distribution analysis and calibration
│   ├── train_and_predict.py               # Stratified 5-fold CV training pipeline
│   └── train_full_and_predict.py          # Full dataset retraining with test density scaling
├── SYNORA_PINNACLE_COLAB_RUNNER.ipynb     # Standalone Google Colab runner for 1-click reproduction
├── colab_pipeline.py                      # Production training pipeline for cloud execution
├── colab_pipeline_fixed.py                # Pipeline with test density normalization
├── colab_pipeline_v5_top1.py              # Fast GPU pipeline reaching 0.4911 milestone
├── colab_pipeline_v6_pinnacle.py          # 15-model bagged GPU ensemble (0.5005 milestone)
├── colab_pipeline_v7_resilient.py         # Resilient checkpointed pipeline with disk serialization
├── colab_cloud_runner.py                  # Headless cloud orchestration runner
├── build_aligned_submission.py            # Timestamp-aligned submission generator & 5-point validator
├── leaderboard_harness.py                 # 5-point submission verification harness
├── deep_eda.py                            # Exploratory data analysis across 1.82M transactions
├── eval_holdout_clean.py                  # September holdout cross-validation evaluator
├── inspect_confusion.py                   # Multi-class confusion matrix forensics
├── inspect_missed.py                      # Recall starvation error diagnostic
├── test_tri_engine.py                     # Tri-engine post-processing test suite
├── test_entity_sharing.py                 # Device and account entity co-occurrence tests
├── test_graph_propagation.py              # Transductive label diffusion verification
├── test_targeted_ring_recovery.py         # Ring recovery threshold exploration
├── test_meta_and_thresholds.py            # Meta-learner and boundary tuning tests
├── test_balanced_subsampling.py           # Class-balanced subsampling experiments
├── test_confidence_amplification.py       # Probability confidence margin amplification
├── submissions/                           # Submission directory
│   └── submission_synora_resilient_aligned_peak.csv  # Final verified peak submission
├── requirements.txt                       # Python dependencies
└── README.md                              # Comprehensive project documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/atharva-awade/Team-Synora-ML-Gradient-Wars-2026.git
cd Team-Synora-ML-Gradient-Wars-2026

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### Option A: 1-Click Google Colab Runner (Recommended)
1. Open `SYNORA_PINNACLE_COLAB_RUNNER.ipynb` in [Google Colab](https://colab.research.google.com).
2. Set runtime type to **GPU** (`T4`, `V100`, or `A100`).
3. Upload the competition data (`train.csv`, `test.csv`, `accounts.csv`, `merchants.csv`) into `./upi-fraud-rings/` or link via Google Drive.
4. Execute **Runtime > Run all**. The notebook will train the models, compute out-of-fold validation scores, and automatically export the submission.

### Option B: Local / Dedicated Cloud Execution
```bash
# 1. Run exploratory data analysis and validation
python deep_eda.py

# 2. Execute full resilient pipeline with 5-fold CV and GPU acceleration
python colab_pipeline_v7_resilient.py

# 3. Generate timestamp-aligned, validated submission
python build_aligned_submission.py
```

---

## 📊 Validation & Leaderboard Progression

| Phase / Submission | Strategy / Innovations | Local CV (Macro-F1) | Public LB (Macro-F1) |
| :--- | :--- | :---: | :---: |
| **Baseline Prototype** | Heuristic quota overrides, single LightGBM | 0.4530 | 0.4514 |
| **Iteration 2** | Purged heuristics, 139 graph & ledger features | 0.4638 | 0.4626 |
| **Synora Omega (v5)** | 50% density scaling ($2.0\times$), high-frequency burst engine | 0.4885 | 0.4911 |
| **Synora Pinnacle (v6)** | 15-model 3-stage stacked GPU ensemble | 0.4992 | **0.5005** |
| **Resilient Aligned Peak** | Holdout-calibrated multipliers, strict ID alignment, dual-stage blend | **0.4796 (Holdout)** | **Peak Submission** |

---

## 🔒 Verification & Compliance Harness

All generated submissions pass our strict 5-point automated verification harness (`leaderboard_harness.py`):
- **Exact Row Count:** Exactly 260,582 test rows.
- **Header & Column Format:** Matches required `id,label` schema.
- **ID Integrity:** 100% bijective match with raw `test.csv` transaction IDs.
- **Zero Nulls:** No NaN or missing entries.
- **Plausible Distribution:** Every predicted class support falls within expected empirical bounds.

---

## 👥 Team

- **Atharva Awade** ([@atharva-awade](https://github.com/atharva-awade)) — *Team Synora*
- **Competition:** ML Gradient Wars 2026 (Hack4Hack)
