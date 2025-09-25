"""
Core evaluation classes for FraudBench
This is your existing code, cleaned up and packaged
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CommonFeatureMapper:
    """Maps different fraud datasets to a common feature representation - NO DATA LEAKAGE"""
    
    COMMON_FEATURES = [
        'normalized_amount',        
        'hour_of_day',             
        'day_of_week',             
        'amount_zscore',           
        'user_transaction_freq',   
        'amount_velocity',         
        'is_weekend',              
        'is_night',                
    ]
    
    def _calculate_safe_user_stats(self, df_train, df_test):
        """Calculate user stats ONLY from training data to prevent leakage"""
        # Check if user_id exists
        if 'user_id' not in df_train.columns:
            # Create dummy user stats
            df_train = df_train.copy()
            df_test = df_test.copy()
            df_train['user_txn_count'] = 1
            df_train['user_avg_amount'] = df_train['amount']
            df_test['user_txn_count'] = 1
            df_test['user_avg_amount'] = df_test['amount']
            return df_train, df_test
        
        # Calculate stats only on training data
        user_stats_train = df_train.groupby('user_id')['amount'].agg(['count', 'mean']).reset_index()
        user_stats_train.columns = ['user_id', 'user_txn_count', 'user_avg_amount']
        
        # Use global defaults for missing users
        global_txn_count = user_stats_train['user_txn_count'].median()
        global_avg_amount = user_stats_train['user_avg_amount'].median()
        
        # Merge with train set
        df_train_merged = df_train.merge(user_stats_train, on='user_id', how='left')
        df_train_merged['user_txn_count'] = df_train_merged['user_txn_count'].fillna(global_txn_count)
        df_train_merged['user_avg_amount'] = df_train_merged['user_avg_amount'].fillna(global_avg_amount)
        
        # Merge with test set (users not in train get global defaults)
        df_test_merged = df_test.merge(user_stats_train, on='user_id', how='left')
        df_test_merged['user_txn_count'] = df_test_merged['user_txn_count'].fillna(global_txn_count)
        df_test_merged['user_avg_amount'] = df_test_merged['user_avg_amount'].fillna(global_avg_amount)
        
        return df_train_merged, df_test_merged
    
    def map_to_common_features(self, df_train, df_test, dataset_name):
        """Map train and test splits to common features WITHOUT data leakage"""
        
        # First, calculate user statistics safely
        df_train_stats, df_test_stats = self._calculate_safe_user_stats(df_train, df_test)
        
        # Calculate amount statistics ONLY from training data
        train_amount_stats = {
            'median': df_train_stats['amount'].median(),
            'mean': df_train_stats['amount'].mean(),
            'std': df_train_stats['amount'].std()
        }
        
        # Create features for both train and test using ONLY training statistics
        train_common = self._create_common_features(df_train_stats, dataset_name, train_amount_stats)
        test_common = self._create_common_features(df_test_stats, dataset_name, train_amount_stats)
        
        return train_common, test_common
    
    def _create_common_features(self, df, dataset_name, amount_stats=None):
        """Create common features from a single dataset"""
        common = pd.DataFrame()
        
        # Amount features - use provided stats or calculate from current df
        if amount_stats is not None:
            # Use pre-calculated statistics (for test set)
            common['normalized_amount'] = df['amount'] / (amount_stats['median'] + 1e-6)
            common['amount_zscore'] = (df['amount'] - amount_stats['mean']) / (amount_stats['std'] + 1e-6)
        else:
            # Calculate from current dataframe (for training set or cross-domain)
            common['normalized_amount'] = df['amount'] / (df['amount'].median() + 1e-6)
            common['amount_zscore'] = (df['amount'] - df['amount'].mean()) / (df['amount'].std() + 1e-6)
        
        # Temporal features
        if 'timestamp' in df.columns:
            if dataset_name == 'ieee_cis':
                hours_since_start = df['timestamp'] / 3600
                common['hour_of_day'] = hours_since_start % 24
                common['day_of_week'] = (hours_since_start // 24) % 7
                
            elif dataset_name == 'paysim':
                common['hour_of_day'] = df['timestamp'] % 24
                common['day_of_week'] = (df['timestamp'] // 24) % 7
                
            elif dataset_name == 'european_cc':
                hours = (df['timestamp'] / 3600) % 24
                common['hour_of_day'] = hours
                common['day_of_week'] = ((df['timestamp'] // 86400) % 7)
        else:
            # Default temporal features
            common['hour_of_day'] = 12
            common['day_of_week'] = 3
        
        # Derived temporal features
        common['is_weekend'] = (common['day_of_week'] >= 5).astype(int)
        common['is_night'] = ((common['hour_of_day'] >= 22) | (common['hour_of_day'] <= 6)).astype(int)
        
        # User behavior features (using the safely calculated stats)
        if 'user_txn_count' in df.columns:
            common['user_transaction_freq'] = np.log1p(df['user_txn_count'])
            common['amount_velocity'] = df['amount'] / (df.get('user_avg_amount', df['amount'].mean()) + 1e-6)
        else:
            common['user_transaction_freq'] = 0
            common['amount_velocity'] = 1
        
        return common[self.COMMON_FEATURES].fillna(0)

class BootstrapAnalyzer:
    """Bootstrap statistical analysis for fraud detection metrics"""
    
    def __init__(self, n_bootstrap=1000, alpha=0.05, random_state=42):
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state
        self.max_sample_size = 50000  # Cap for memory efficiency
    
    def bootstrap_metric(self, y_true, y_pred_proba, metric_func):
        """Bootstrap confidence interval for a single metric"""
        bootstrap_scores = []
        n_samples = len(y_true)
        
        # Ensure we have numpy arrays
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        
        # For very large datasets, use subsampling
        if n_samples > self.max_sample_size:
            from sklearn.model_selection import train_test_split
            _, y_true, _, y_pred_proba = train_test_split(
                y_true, y_pred_proba, 
                test_size=min(self.max_sample_size, n_samples),
                stratify=y_true if len(np.unique(y_true)) > 1 else None,
                random_state=self.random_state
            )
            n_samples = len(y_true)
        
        # Create a local random generator
        rng = np.random.RandomState(self.random_state)
        
        # Reduce bootstrap iterations for large datasets
        n_bootstrap_adj = min(self.n_bootstrap, 500) if n_samples > 50000 else self.n_bootstrap
        
        for i in range(n_bootstrap_adj):
            try:
                # Bootstrap sample
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                y_true_boot = y_true[indices]
                y_pred_boot = y_pred_proba[indices]
                
                # Ensure we have both classes in bootstrap sample
                if len(np.unique(y_true_boot)) < 2:
                    continue
                
                score = metric_func(y_true_boot, y_pred_boot)
                bootstrap_scores.append(score)
                
            except Exception as e:
                # Skip failed iterations
                continue
        
        bootstrap_scores = np.array(bootstrap_scores)
        
        if len(bootstrap_scores) == 0:
            return {
                'mean': metric_func(y_true, y_pred_proba),
                'std': 0.0,
                'ci_lower': metric_func(y_true, y_pred_proba),
                'ci_upper': metric_func(y_true, y_pred_proba),
                'bootstrap_scores': np.array([metric_func(y_true, y_pred_proba)])
            }
        
        # Calculate statistics
        mean_score = np.mean(bootstrap_scores)
        std_score = np.std(bootstrap_scores)
        lower_ci = np.percentile(bootstrap_scores, 100 * self.alpha/2)
        upper_ci = np.percentile(bootstrap_scores, 100 * (1 - self.alpha/2))
        
        return {
            'mean': mean_score,
            'std': std_score,
            'ci_lower': lower_ci,
            'ci_upper': upper_ci,
            'bootstrap_scores': bootstrap_scores
        }
    
    def compare_independent_distributions(self, scores_a, scores_b):
        """Compare two independent bootstrap distributions"""
        # Welch's t-test
        statistic, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)
        
        # Calculate effect size
        mean_a = np.mean(scores_a)
        mean_b = np.mean(scores_b)
        mean_difference = mean_a - mean_b
        
        # Cohen's d
        std_a, std_b = np.std(scores_a, ddof=1), np.std(scores_b, ddof=1)
        n_a, n_b = len(scores_a), len(scores_b)
        pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        
        cohens_d = mean_difference / pooled_std if pooled_std > 0 else 0
        
        # Effect size interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            effect_size = "negligible"
        elif abs_d < 0.5:
            effect_size = "small"
        elif abs_d < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        return {
            't_statistic': statistic,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'effect_size': effect_size,
            'significant': p_value < 0.05,
            'mean_difference': mean_difference,
            'mean_a': mean_a,
            'mean_b': mean_b
        }