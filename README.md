# FraudBench: A Benchmark for Cross-Domain Fraud Detection

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

FraudBench is a comprehensive benchmark for evaluating cross-domain fraud detection models. It reveals the catastrophic failure of fraud detection systems when deployed across different domains, with performance degradation up to 98.7% in AUC-PR.

### Key Contributions
- **Standardized Benchmark**: Evaluation across 3 datasets (IEEE-CIS, PaySim, European CC) and 6 transfer scenarios
- **Novel Transferability Prediction**: Predicts cross-domain performance WITHOUT training models (97.8% computation savings)
- **Statistical Rigor**: Bootstrap confidence intervals (1,000 samples) for all metrics
- **Practical Insights**: Identifies which domain transfers will fail before deployment

## Project Structure

```
fraudbench/
├── fraudbench.py           # Main benchmark interface
├── data_loaders.py         # Dataset loading and preprocessing
├── evaluator_fixed.py      # Core evaluation classes (metrics, feature mapping)
├── transferability.py      # Novel transferability prediction framework
├── demo_novelty.py         # Demonstration of novel contributions
├── run_all_experiments.py  # Reproduce paper results
├── requirements.txt        # Python dependencies
└── data/
    └── README.md          # Dataset download instructions
```

## File Descriptions

### Core Modules

**`fraudbench.py`** - Main benchmark interface
- `FraudBench` class: Primary API for model evaluation
- Methods for cross-domain evaluation, result saving, and baseline comparison
- Manages data caching and standardized evaluation protocol

**`data_loaders.py`** - Dataset management
- `FraudBenchDataLoader` class: Handles all dataset operations
- Automatic downloading for PaySim and European CC datasets
- Fallback synthetic data generation for testing

**`evaluator_fixed.py`** - Evaluation infrastructure
- `CommonFeatureMapper`: Maps heterogeneous datasets to 8 common features
- `BootstrapAnalyzer`: Statistical validation with confidence intervals
- Prevents data leakage between train/test splits

**`transferability.py`** - Novel contribution
- `TransferabilityPredictor`: Predicts cross-domain success WITHOUT training
- Computes domain similarity using MMD, KL divergence, and statistical tests
- Meta-learning framework to predict AUC-PR from domain characteristics

### Scripts

**`demo_novelty.py`** - Showcase novel contributions
- Demonstrates transferability prediction
- Shows 97.8% computation savings
- Risk assessment for all domain pairs

**`run_all_experiments.py`** - Reproduce paper results
- Runs all baseline models (LightGBM, MLP, IsolationForest)
- Tests all 6 cross-domain scenarios
- Generates LaTeX tables for paper

## Installation

### Requirements
- Python 3.11+
- 2GB disk space for datasets
- 8GB RAM recommended

### Setup

```
# Clone repository
git clone https://github.com/yourusername/fraudbench.git
cd fraudbench

# Install dependencies
pip install -r requirements.txt

# Create data directory
mkdir fraudbench_data
```

## Quick Start

### Basic Usage

```python
from fraudbench import FraudBench
from sklearn.ensemble import RandomForestClassifier

# Initialize benchmark
bench = FraudBench(data_dir='./fraudbench_data')
bench.prepare_datasets()  # Downloads data automatically

# Evaluate your model
model = RandomForestClassifier(n_estimators=100)
results = bench.evaluate(model, source='paysim', target='ieee_cis')

# View results
print(f"AUC-PR: {results['metrics']['auc_pr']['mean']:.3f}")
print(f"95% CI: [{results['metrics']['auc_pr']['ci_lower']:.3f}, "
      f"{results['metrics']['auc_pr']['ci_upper']:.3f}]")
```

### Transferability Prediction (Novel Feature)

```python
from transferability import TransferabilityPredictor

# Initialize predictor
predictor = TransferabilityPredictor()

# Load datasets
source_X, source_y = bench.load_processed_dataset('paysim')
target_X, target_y = bench.load_processed_dataset('ieee_cis')

# Predict transfer success WITHOUT training
similarity = predictor.compute_domain_similarity(
    source_X, source_y, target_X, target_y
)

print(f"Transferability Score: {similarity['overall_transferability']:.3f}")
if similarity['overall_transferability'] < 0.5:
    print("⚠️ High risk of failure - avoid deployment")
```

## Datasets

### Required Datasets
- **IEEE-CIS (~1.2GB)** - Manual download from Kaggle
- **PaySim (~470MB)** - Automatically downloaded
- **European CC (~144MB)** - Automatically downloaded

See `data/README.md` for detailed download instructions.

## Key Results

### Cross-Domain Performance Degradation

| Transfer | LightGBM AUC-PR | Degradation |
|----------|----------------|-------------|
| PaySim → IEEE-CIS | 0.061 | -61.0% |
| PaySim → European | 0.014 | -91.1% |
| IEEE-CIS → European | 0.003 | -98.7% |

### Transferability Predictions

| Domain Pair | Score | Risk Level |
|-------------|-------|------------|
| PaySim ↔ IEEE-CIS | 0.487 | High |
| PaySim ↔ European | 0.766 | Low |
| IEEE-CIS ↔ European | 0.510 | Medium |

## Reproducing Paper Results

```
# Run all experiments (takes ~45 minutes)
python run_all_experiments.py

# Demonstrate novel contributions (takes <1 minute)
python demo_novelty.py
```

## Citation

If you use FraudBench in your research, please cite:

```bibtex
@inproceedings{fraudbench2025,
  title={FraudBench: A Benchmark for Cross-Domain Fraud Detection},
  author={[Authors]},
  booktitle={KDD '25: The 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2025}
}
```

## Features

### Common Feature Space
All datasets are mapped to 8 standardized features:
- **normalized_amount**: Transaction amount relative to median
- **hour_of_day**: Hour extracted from timestamp
- **day_of_week**: Day of week (0-6)
- **amount_zscore**: Standardized amount
- **user_transaction_freq**: Log of user's transaction count
- **amount_velocity**: Ratio to user's average amount
- **is_weekend**: Binary weekend indicator
- **is_night**: Binary night-time indicator

### Evaluation Metrics
- **Primary**: Area Under Precision-Recall Curve (AUC-PR)
- **Secondary**: Area Under ROC Curve (AUC-ROC)
- **Statistical**: 95% bootstrap confidence intervals
- **Operational**: Fraud rate, sample sizes

### Models Tested
- **Tree-based**: LightGBM, RandomForest
- **Neural**: Multi-Layer Perceptron
- **Unsupervised**: Isolation Forest

## Contributions

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with your improvements

## License

MIT License - see LICENSE file for details

## Acknowledgments

- IEEE-CIS dataset provided by Vesta Corporation
- PaySim synthetic data generator
- European Credit Card dataset from MLG Université Libre de Bruxelles
