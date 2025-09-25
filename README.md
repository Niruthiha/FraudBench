# FraudBench Dataset Instructions

## Required Datasets

### 1. IEEE-CIS Fraud Detection 
- **Source**: [Kaggle Competition](https://www.kaggle.com/c/ieee-fraud-detection/data)
- **Files needed**:
  - `train_transaction.csv`
  - `test_transaction.csv`
  - `train_identity.csv`
  - `test_identity.csv`
- **Download**: Requires a free Kaggle account

### 2. PaySim Synthetic Dataset
- Automatically downloaded via `FraudBench.prepare_datasets()`

### 3. European Credit Card Dataset
- **Source**: Public MLG dataset
- Automatically downloaded by the data loader

---

## Setup Instructions

### 1. Create a data directory   
    ```bash
    mkdir fraudbench_data



## 2. Download IEEE-CIS Dataset

### Option 1: Manual Download

1. Visit: [https://www.kaggle.com/c/ieee-fraud-detection/data](https://www.kaggle.com/c/ieee-fraud-detection/data)
2. Download the required files:
   - `train_transaction.csv`
   - `test_transaction.csv`
   - `train_identity.csv`
   - `test_identity.csv`
3. Move or extract them into the `fraudbench_data/` directory

---

### Option 2: Using Kaggle API

    ```bash
        pip install kaggle

### Authenticate with Kaggle (requires API token)
kaggle competitions download -c ieee-fraud-detection

#### Unzip into data directory
unzip ieee-fraud-detection.zip -d fraudbench_data/


### Option 3: Load and Prepare Datasets in Python

      ```bash
      from fraudbench import FraudBench

bench = FraudBench(data_dir='./fraudbench_data')
bench.prepare_datasets()  # Downloads PaySim and European Credit Card datasets



