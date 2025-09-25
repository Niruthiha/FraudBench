# FraudBench Dataset Instructions

## Required Datasets

### 1. IEEE-CIS Fraud Detection 
- Source: Kaggle Competition
- URL: https://www.kaggle.com/c/ieee-fraud-detection/data
- Files needed:
  - `train_transaction.csv`
  - `test_transaction.csv`
  - `train_identity.csv`
  - `test_identity.csv`
- Download: Requires Kaggle account (free)

### 2. PaySim Synthetic Dataset 
- Source: Public synthetic dataset
- The data loader will automatically download this

### 3. European Credit Card 
- Source: Public MLG dataset
- The data loader will automatically download this

## Setup Instructions

1. Create data directory:
  ```bash
  mkdir fraudbench_data

-----

2. # Option 1: Manual download
# Go to https://www.kaggle.com/c/ieee-fraud-detection/data
# Download and extract to fraudbench_data/

# Option 2: Using Kaggle API
pip install kaggle
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d fraudbench_data/

3. from fraudbench import FraudBench
bench = FraudBench(data_dir='./fraudbench_data')
bench.prepare_datasets()  # Downloads PaySim and European CC automatically

