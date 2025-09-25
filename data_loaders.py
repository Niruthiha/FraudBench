"""
FraudBench Data Loaders
Handles loading of fraud detection datasets from your actual files
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

class FraudBenchDataLoader:
    """Load and prepare fraud detection datasets"""
    
    def __init__(self, data_dir='./fraudbench_data'):
        self.data_dir = Path(data_dir)
        
        # Check if data directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} not found!")
        
        print(f"Data directory: {self.data_dir.absolute()}")
        
    def load_dataset(self, dataset_name, sample_size=None):
        """Load a specific dataset"""
        
        if dataset_name == 'ieee_cis':
            return self._load_ieee_cis(sample_size)
        elif dataset_name == 'paysim':
            return self._load_paysim(sample_size)
        elif dataset_name == 'european_cc':
            return self._load_european_cc(sample_size)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _load_ieee_cis(self, sample_size=None):
        """Load IEEE-CIS dataset from your files"""
        print("Loading IEEE-CIS dataset...")
        
        # Your actual file names
        train_trans_file = self.data_dir / 'train_transaction.csv'
        test_trans_file = self.data_dir / 'test_transaction.csv'
        train_id_file = self.data_dir / 'train_identity.csv'
        test_id_file = self.data_dir / 'test_identity.csv'
        
        if not train_trans_file.exists():
            raise FileNotFoundError(f"IEEE-CIS train file not found at {train_trans_file}")
        
        # Load transaction data
        print("  Loading train transactions...")
        df_train = pd.read_csv(train_trans_file)
        
        # For cross-domain, we use all data as one dataset
        if test_trans_file.exists():
            print("  Loading test transactions...")
            df_test = pd.read_csv(test_trans_file)
            # Test data doesn't have labels, so we'll only use train for now
            df = df_train
        else:
            df = df_train
        
        # Sample if needed for memory
        if sample_size and len(df) > sample_size:
            print(f"  Sampling {sample_size} from {len(df)} transactions...")
            df = df.sample(n=sample_size, random_state=42)
        elif len(df) > 500000:  # Default limit for memory
            print(f"  Sampling 500000 from {len(df)} transactions for memory efficiency...")
            df = df.sample(n=500000, random_state=42)
        
        # Prepare features
        X = pd.DataFrame()
        
        # User ID (card1 is the main card identifier)
        X['user_id'] = df['card1'].fillna(-1).astype(str)
        
        # Amount
        X['amount'] = df['TransactionAmt'].fillna(0)
        
        # Timestamp (TransactionDT is seconds from reference)
        X['timestamp'] = df['TransactionDT'].fillna(0)
        
        # Add other useful features if available
        feature_cols = ['ProductCD', 'card2', 'card3', 'card4', 'card5', 'card6', 
                       'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']
        
        for col in feature_cols:
            if col in df.columns:
                X[col] = df[col]
        
        # Target variable
        y = df['isFraud']
        
        print(f"  Loaded {len(X)} transactions, {y.mean():.2%} fraud rate")
        return X, y
    
    def _load_paysim(self, sample_size=None):
        """Load PaySim dataset from your file"""
        print("Loading PaySim dataset...")
        
        # Your actual file name
        paysim_file = self.data_dir / 'paysim_dataset.csv'
        
        if not paysim_file.exists():
            raise FileNotFoundError(f"PaySim file not found at {paysim_file}")
        
        # Load data
        print("  Reading PaySim CSV...")
        df = pd.read_csv(paysim_file)
        
        # Check column names (PaySim might have different column names)
        print(f"  PaySim columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
        
        # Sample if needed (PaySim is large)
        if sample_size and len(df) > sample_size:
            print(f"  Sampling {sample_size} from {len(df)} transactions...")
            df = df.sample(n=sample_size, random_state=42)
        elif len(df) > 500000:  # Default sampling for memory
            print(f"  Sampling 500000 from {len(df)} transactions for memory efficiency...")
            df = df.sample(n=500000, random_state=42)
        
        # Prepare features - adjust column names based on actual PaySim columns
        X = pd.DataFrame()
        
        # Common PaySim column names
        if 'nameOrig' in df.columns:
            X['user_id'] = df['nameOrig'].astype(str)
        elif 'customer' in df.columns:
            X['user_id'] = df['customer'].astype(str)
        else:
            # Create synthetic user IDs
            X['user_id'] = np.arange(len(df)).astype(str)
        
        # Amount
        if 'amount' in df.columns:
            X['amount'] = df['amount']
        elif 'Amount' in df.columns:
            X['amount'] = df['Amount']
        else:
            X['amount'] = df.iloc[:, df.select_dtypes(include=[np.number]).columns[0]]  # First numeric column
        
        # Timestamp
        if 'step' in df.columns:
            X['timestamp'] = df['step']  # Hours from start
        elif 'hour' in df.columns:
            X['timestamp'] = df['hour']
        else:
            X['timestamp'] = np.arange(len(df))  # Synthetic timestamps
        
        # Transaction type if available
        if 'type' in df.columns:
            X['type'] = df['type']
        
        # Balance features if available
        for col in ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']:
            if col in df.columns:
                X[col] = df[col]
        
        # Target variable
        if 'isFraud' in df.columns:
            y = df['isFraud']
        elif 'fraud' in df.columns:
            y = df['fraud']
        elif 'is_fraud' in df.columns:
            y = df['is_fraud']
        else:
            # Look for any column with fraud in the name
            fraud_cols = [col for col in df.columns if 'fraud' in col.lower()]
            if fraud_cols:
                y = df[fraud_cols[0]]
            else:
                raise ValueError(f"Could not find fraud label column in PaySim. Columns: {df.columns.tolist()}")
        
        print(f"  Loaded {len(X)} transactions, {y.mean():.2%} fraud rate")
        return X, y
    
    def _load_european_cc(self, sample_size=None):
        """Load European Credit Card dataset from your file"""
        print("Loading European Credit Card dataset...")
        
        # Your actual file name
        cc_file = self.data_dir / 'creditcard.csv'
        
        if not cc_file.exists():
            raise FileNotFoundError(f"European CC file not found at {cc_file}")
        
        # Load data
        print("  Reading creditcard CSV...")
        df = pd.read_csv(cc_file)
        
        # Sample if needed
        if sample_size and len(df) > sample_size:
            print(f"  Sampling {sample_size} from {len(df)} transactions...")
            df = df.sample(n=sample_size, random_state=42)
        
        # Prepare features
        X = pd.DataFrame()
        
        # Create synthetic user IDs (this dataset doesn't have user IDs)
        # We'll create them based on transaction patterns
        np.random.seed(42)
        n_users = max(100, len(df) // 100)
        X['user_id'] = np.random.choice(n_users, size=len(df)).astype(str)
        
        # Amount
        X['amount'] = df['Amount']
        
        # Timestamp (Time is seconds from first transaction)
        X['timestamp'] = df['Time']
        
        # Include PCA components V1-V28
        pca_cols = [f'V{i}' for i in range(1, 29)]
        for col in pca_cols:
            if col in df.columns:
                X[col] = df[col]
        
        # Target variable
        y = df['Class']
        
        print(f"  Loaded {len(X)} transactions, {y.mean():.2%} fraud rate")
        return X, y
    
    def load_all_datasets(self, sample_size=None):
        """Load all available datasets"""
        datasets = {}
        
        for dataset_name in ['ieee_cis', 'paysim', 'european_cc']:
            try:
                X, y = self.load_dataset(dataset_name, sample_size)
                datasets[dataset_name] = (X, y)
            except Exception as e:
                print(f"Failed to load {dataset_name}: {e}")
        
        return datasets
    
    def check_data_files(self):
        """Check which data files are available"""
        print("\nChecking data files in", self.data_dir.absolute())
        print("-" * 50)
        
        expected_files = {
            'IEEE-CIS': ['train_transaction.csv', 'test_transaction.csv', 
                        'train_identity.csv', 'test_identity.csv'],
            'PaySim': ['paysim_dataset.csv'],
            'European CC': ['creditcard.csv']
        }
        
        for dataset, files in expected_files.items():
            print(f"\n{dataset}:")
            for file in files:
                path = self.data_dir / file
                if path.exists():
                    size_mb = path.stat().st_size / (1024 * 1024)
                    print(f"  ✓ {file} ({size_mb:.1f} MB)")
                else:
                    print(f"  ✗ {file} (NOT FOUND)")
        
        print("-" * 50)

# Test function
if __name__ == "__main__":
    print("Testing FraudBench Data Loader...")
    loader = FraudBenchDataLoader('/home/paulj/niru/fraudbench/fraudbench_data')
    
    # Check files
    loader.check_data_files()
    
    # Try loading each dataset
    print("\nLoading datasets...")
    datasets = loader.load_all_datasets(sample_size=10000)  # Small sample for testing
    
    for name, (X, y) in datasets.items():
        print(f"\n{name}:")
        print(f"  Shape: {X.shape}")
        print(f"  Fraud rate: {y.mean():.4%}")
        print(f"  Columns: {X.columns.tolist()[:5]}...")  # First 5 columns