"""
FraudBench: A Benchmark for Cross-Domain Fraud Detection
Main module providing the benchmark interface
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
from datetime import datetime
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import your existing classes
from evaluator_fixed import CommonFeatureMapper, BootstrapAnalyzer

class FraudBench:
    """
    Main interface for the FraudBench benchmark.
    
    Example usage:
        from fraudbench import FraudBench
        from sklearn.ensemble import RandomForestClassifier
        
        bench = FraudBench()
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        results = bench.evaluate(model, source='paysim', target='ieee_cis')
        print(f"AUC-PR: {results['auc_pr']['mean']:.3f} [{results['auc_pr']['ci_lower']:.3f}, {results['auc_pr']['ci_upper']:.3f}]")
    """
    
    def __init__(self, data_dir='./fraudbench_data', cache_dir='./fraudbench_cache'):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.mapper = CommonFeatureMapper()
        self.bootstrap_analyzer = BootstrapAnalyzer(n_bootstrap=1000)
        
        # Define standard scenarios
        self.scenarios = {
            'geographic_expansion': {
                'source': 'ieee_cis',
                'target': 'european_cc',
                'description': 'US e-commerce to European credit cards'
            },
            'payment_evolution': {
                'source': 'european_cc', 
                'target': 'paysim',
                'description': 'Traditional cards to mobile payments'
            },
            'synthetic_to_real': {
                'source': 'paysim',
                'target': 'ieee_cis',
                'description': 'Synthetic to real-world deployment'
            },
            'real_to_synthetic': {
                'source': 'ieee_cis',
                'target': 'paysim',
                'description': 'Real-world to synthetic validation'
            },
            'temporal_shift': {
                'source': 'european_cc',
                'target': 'ieee_cis',
                'description': '2013 to 2019 fraud patterns'
            },
            'cross_platform': {
                'source': 'paysim',
                'target': 'european_cc',
                'description': 'Mobile money to credit cards'
            }
        }
        
        # Baseline results for comparison
        self.baseline_results = {
            ('paysim', 'ieee_cis'): {
                'lightgbm': {'auc_pr': 0.061, 'auc_roc': 0.551},
                'mlp': {'auc_pr': 0.034, 'auc_roc': 0.493},
                'isolation_forest': {'auc_pr': 0.056, 'auc_roc': 0.527}
            },
            ('ieee_cis', 'paysim'): {
                'lightgbm': {'auc_pr': 0.045, 'auc_roc': 0.755},
                'mlp': {'auc_pr': 0.124, 'auc_roc': 0.672},
                'isolation_forest': {'auc_pr': 0.022, 'auc_roc': 0.810}
            },
            ('ieee_cis', 'european_cc'): {
                'lightgbm': {'auc_pr': 0.003, 'auc_roc': 0.642},
                'mlp': {'auc_pr': 0.002, 'auc_roc': 0.566},
                'isolation_forest': {'auc_pr': 0.003, 'auc_roc': 0.620}
            },
            ('european_cc', 'ieee_cis'): {
                'lightgbm': {'auc_pr': 0.041, 'auc_roc': 0.604},
                'mlp': {'auc_pr': 0.110, 'auc_roc': 0.541},
                'isolation_forest': {'auc_pr': 0.082, 'auc_roc': 0.543}
            },
            ('paysim', 'european_cc'): {
                'lightgbm': {'auc_pr': 0.014, 'auc_roc': 0.593},
                'mlp': {'auc_pr': 0.002, 'auc_roc': 0.477},
                'isolation_forest': {'auc_pr': 0.004, 'auc_roc': 0.546}
            },
            ('european_cc', 'paysim'): {
                'lightgbm': {'auc_pr': 0.131, 'auc_roc': 0.746},
                'mlp': {'auc_pr': 0.203, 'auc_roc': 0.919},
                'isolation_forest': {'auc_pr': 0.045, 'auc_roc': 0.823}
            }
        }
    
    def prepare_datasets(self, force_reload=False):
        """
        Prepare and cache all datasets with common features.
        Run this once before evaluation.
        """
        from data_loaders import FraudBenchDataLoader
        
        print("Preparing FraudBench datasets...")
        loader = FraudBenchDataLoader(self.data_dir)
        
        for dataset_name in ['ieee_cis', 'paysim', 'european_cc']:
            cache_file = self.cache_dir / f'{dataset_name}_processed.pkl'
            
            if cache_file.exists() and not force_reload:
                print(f"  {dataset_name}: Using cached version")
                continue
            
            print(f"  {dataset_name}: Loading and processing...")
            try:
                X, y = loader.load_dataset(dataset_name)
                
                # Apply common features
                X_with_stats = self.mapper._calculate_safe_user_stats(X, X)[0]
                X_common = self.mapper._create_common_features(X_with_stats, dataset_name)
                
                # Save processed data
                with open(cache_file, 'wb') as f:
                    pickle.dump({'X': X_common, 'y': y}, f)
                
                print(f"    Processed {len(X_common)} samples, {X_common.shape[1]} features")
                
            except Exception as e:
                print(f"    Error: {e}")
    
    def load_processed_dataset(self, dataset_name):
        """Load preprocessed dataset from cache"""
        cache_file = self.cache_dir / f'{dataset_name}_processed.pkl'
        
        if not cache_file.exists():
            raise FileNotFoundError(
                f"Dataset {dataset_name} not found. Run prepare_datasets() first."
            )
        
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        
        return data['X'], data['y']
    
    def evaluate(self, model, source, target, return_predictions=False):
        """
        Evaluate a model on cross-domain transfer.
        
        Args:
            model: A scikit-learn compatible model with fit() and predict_proba()
            source: Name of source dataset ('ieee_cis', 'paysim', 'european_cc')
            target: Name of target dataset
            return_predictions: If True, return predictions along with metrics
        
        Returns:
            dict: Evaluation results including metrics with confidence intervals
        """
        print(f"\nEvaluating {model.__class__.__name__}: {source} → {target}")
        
        # Load datasets
        X_source, y_source = self.load_processed_dataset(source)
        X_target, y_target = self.load_processed_dataset(target)
        
        print(f"  Source: {len(X_source)} samples, {y_source.mean():.3%} fraud")
        print(f"  Target: {len(X_target)} samples, {y_target.mean():.3%} fraud")
        
        # Train on source
        print("  Training on source...")
        model.fit(X_source, y_source)
        
        # Predict on target
        print("  Evaluating on target...")
        if hasattr(model, 'predict_proba'):
            y_pred = model.predict_proba(X_target)[:, 1]
        else:
            # For models like IsolationForest
            scores = model.decision_function(X_target)
            y_pred = (scores - scores.min()) / (scores.max() - scores.min())
        
        # Compute metrics
        print("  Computing metrics with bootstrap CI...")
        metrics = self._compute_metrics_with_bootstrap(y_target, y_pred)
        
        # Compare to baselines
        baseline = self.baseline_results.get((source, target), {})
        
        results = {
            'model': model.__class__.__name__,
            'source': source,
            'target': target,
            'metrics': metrics,
            'baseline_comparison': baseline,
            'timestamp': datetime.now().isoformat()
        }
        
        if return_predictions:
            results['predictions'] = {
                'y_true': y_target.values if hasattr(y_target, 'values') else y_target,
                'y_pred': y_pred
            }
        
        # Print summary
        print(f"\n  Results:")
        print(f"    AUC-ROC: {metrics['auc_roc']['mean']:.3f} [{metrics['auc_roc']['ci_lower']:.3f}, {metrics['auc_roc']['ci_upper']:.3f}]")
        print(f"    AUC-PR:  {metrics['auc_pr']['mean']:.3f} [{metrics['auc_pr']['ci_lower']:.3f}, {metrics['auc_pr']['ci_upper']:.3f}]")
        
        if baseline:
            print(f"\n  Baseline comparison:")
            for model_name, scores in baseline.items():
                print(f"    {model_name}: AUC-PR={scores['auc_pr']:.3f}, AUC-ROC={scores['auc_roc']:.3f}")
        
        return results
    
    def evaluate_scenario(self, model, scenario_name):
        """
        Evaluate a model on a named scenario.
        
        Args:
            model: A scikit-learn compatible model
            scenario_name: One of the predefined scenarios
        
        Returns:
            dict: Evaluation results
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}. "
                           f"Available: {list(self.scenarios.keys())}")
        
        scenario = self.scenarios[scenario_name]
        print(f"\nScenario: {scenario_name}")
        print(f"Description: {scenario['description']}")
        
        return self.evaluate(
            model, 
            source=scenario['source'],
            target=scenario['target']
        )
    
    def evaluate_all_scenarios(self, model):
        """Evaluate a model on all scenarios"""
        results = {}
        
        for scenario_name in self.scenarios:
            results[scenario_name] = self.evaluate_scenario(model, scenario_name)
        
        return results
    
    def _compute_metrics_with_bootstrap(self, y_true, y_pred):
        """Compute metrics with bootstrap confidence intervals"""
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Compute point estimates
        try:
            auc_roc = roc_auc_score(y_true, y_pred)
        except:
            auc_roc = 0.5
        
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            auc_pr = auc(recall, precision)
        except:
            auc_pr = y_true.mean()  # Baseline for random classifier
        
        # Bootstrap analysis
        roc_bootstrap = self.bootstrap_analyzer.bootstrap_metric(
            y_true, y_pred, roc_auc_score
        )
        
        def auc_pr_metric(y_t, y_p):
            p, r, _ = precision_recall_curve(y_t, y_p)
            return auc(r, p)
        
        pr_bootstrap = self.bootstrap_analyzer.bootstrap_metric(
            y_true, y_pred, auc_pr_metric
        )
        
        return {
            'auc_roc': {
                'mean': roc_bootstrap['mean'],
                'std': roc_bootstrap['std'],
                'ci_lower': roc_bootstrap['ci_lower'],
                'ci_upper': roc_bootstrap['ci_upper']
            },
            'auc_pr': {
                'mean': pr_bootstrap['mean'],
                'std': pr_bootstrap['std'],
                'ci_lower': pr_bootstrap['ci_lower'],
                'ci_upper': pr_bootstrap['ci_upper']
            },
            'n_samples': len(y_true),
            'fraud_rate': y_true.mean()
        }
    
    def save_results(self, results, filename=None):
        """Save evaluation results to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'fraudbench_results_{timestamp}.json'
        
        output_file = Path(filename)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        results_serializable = convert_numpy(results)
        
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
    
    def generate_leaderboard(self, new_results=None):
        """
        Generate a leaderboard comparing new results to baselines.
        
        Args:
            new_results: Optional dict of new results to include
        
        Returns:
            pd.DataFrame: Leaderboard table
        """
        rows = []
        
        # Add baseline results
        for (source, target), models in self.baseline_results.items():
            for model_name, metrics in models.items():
                rows.append({
                    'Model': model_name,
                    'Source': source,
                    'Target': target,
                    'AUC-PR': metrics['auc_pr'],
                    'AUC-ROC': metrics['auc_roc'],
                    'Type': 'Baseline'
                })
        
        # Add new results if provided
        if new_results:
            if isinstance(new_results, dict) and 'metrics' in new_results:
                # Single result
                rows.append({
                    'Model': new_results['model'],
                    'Source': new_results['source'],
                    'Target': new_results['target'],
                    'AUC-PR': new_results['metrics']['auc_pr']['mean'],
                    'AUC-ROC': new_results['metrics']['auc_roc']['mean'],
                    'Type': 'New'
                })
        
        df = pd.DataFrame(rows)
        
        # Sort by AUC-PR (our primary metric)
        df = df.sort_values('AUC-PR', ascending=False)
        
        return df