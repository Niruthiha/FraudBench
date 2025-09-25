"""
Transferability Prediction Framework for FraudBench
Novel contribution: Predict cross-domain performance BEFORE deployment
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class TransferabilityPredictor:
    """
    Predict if a model will transfer well BEFORE deployment.
    This is a novel contribution that helps practitioners avoid costly failures.
    """
    
    def __init__(self):
        self.meta_model = None
        self.feature_importance = None
        
    def compute_domain_similarity(self, source_X, source_y, target_X, target_y):
        """
        Compute comprehensive similarity metrics between domains.
        Higher scores indicate better expected transfer.
        """
        
        metrics = {}
        
        # 1. Label Shift - how different are fraud rates?
        source_fraud_rate = source_y.mean()
        target_fraud_rate = target_y.mean()
        metrics['label_shift'] = 1.0 - abs(source_fraud_rate - target_fraud_rate) / max(source_fraud_rate, target_fraud_rate)
        
        # 2. Feature Distribution Similarity (per feature)
        feature_similarities = []
        for col in source_X.columns:
            if col in target_X.columns:
                # KS statistic for distribution difference
                ks_stat, _ = stats.ks_2samp(source_X[col], target_X[col])
                feature_similarities.append(1.0 - ks_stat)  # Convert to similarity
        
        metrics['feature_similarity'] = np.mean(feature_similarities) if feature_similarities else 0.0
        
        # 3. Maximum Mean Discrepancy (MMD) - overall distribution difference
        mmd = self._compute_mmd(source_X.values, target_X.values)
        metrics['mmd_similarity'] = 1.0 / (1.0 + mmd)  # Convert to similarity
        
        # 4. Amount Distribution Similarity (critical for fraud)
        if 'normalized_amount' in source_X.columns and 'normalized_amount' in target_X.columns:
            amount_ks, _ = stats.ks_2samp(
                source_X['normalized_amount'], 
                target_X['normalized_amount']
            )
            metrics['amount_similarity'] = 1.0 - amount_ks
        else:
            metrics['amount_similarity'] = 0.5
        
        # 5. Temporal Pattern Similarity
        if 'hour_of_day' in source_X.columns and 'hour_of_day' in target_X.columns:
            # Compare hourly distributions
            source_hours = np.histogram(source_X['hour_of_day'], bins=24, range=(0, 24))[0]
            target_hours = np.histogram(target_X['hour_of_day'], bins=24, range=(0, 24))[0]
            
            # Normalize to probabilities
            source_hours = source_hours / source_hours.sum()
            target_hours = target_hours / target_hours.sum()
            
            # Jensen-Shannon divergence (symmetric KL divergence)
            js_div = jensenshannon(source_hours, target_hours)
            metrics['temporal_similarity'] = 1.0 - js_div
        else:
            metrics['temporal_similarity'] = 0.5
        
        # 6. Class Imbalance Ratio Difference
        source_imbalance = (1 - source_fraud_rate) / (source_fraud_rate + 1e-6)
        target_imbalance = (1 - target_fraud_rate) / (target_fraud_rate + 1e-6)
        imbalance_ratio = min(source_imbalance, target_imbalance) / max(source_imbalance, target_imbalance)
        metrics['imbalance_similarity'] = imbalance_ratio
        
        # 7. Statistical Moments Similarity (mean, std, skew)
        moment_similarities = []
        for col in source_X.select_dtypes(include=[np.number]).columns:
            if col in target_X.columns:
                source_mean = source_X[col].mean()
                target_mean = target_X[col].mean()
                source_std = source_X[col].std()
                target_std = target_X[col].std()
                
                mean_sim = 1.0 - abs(source_mean - target_mean) / (abs(source_mean) + abs(target_mean) + 1e-6)
                std_sim = 1.0 - abs(source_std - target_std) / (source_std + target_std + 1e-6)
                moment_similarities.extend([mean_sim, std_sim])
        
        metrics['moment_similarity'] = np.mean(moment_similarities) if moment_similarities else 0.5
        
        # 8. Compute overall transferability score
        weights = {
            'label_shift': 0.25,
            'feature_similarity': 0.20,
            'mmd_similarity': 0.15,
            'amount_similarity': 0.15,
            'temporal_similarity': 0.10,
            'imbalance_similarity': 0.10,
            'moment_similarity': 0.05
        }
        
        metrics['overall_transferability'] = sum(
            metrics[key] * weight for key, weight in weights.items()
        )
        
        return metrics
    
    def _compute_mmd(self, X, Y, kernel='rbf', gamma=1.0):
        """
        Compute Maximum Mean Discrepancy between two distributions.
        Lower values indicate more similar distributions.
        """
        n = min(len(X), 1000)  # Subsample for efficiency
        m = min(len(Y), 1000)
        
        if len(X) > n:
            idx = np.random.choice(len(X), n, replace=False)
            X = X[idx]
        if len(Y) > m:
            idx = np.random.choice(len(Y), m, replace=False)
            Y = Y[idx]
        
        # RBF kernel
        def rbf_kernel(X, Y, gamma):
            dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
            return np.exp(-gamma * dists)
        
        K_XX = rbf_kernel(X, X, gamma)
        K_YY = rbf_kernel(Y, Y, gamma)
        K_XY = rbf_kernel(X, Y, gamma)
        
        mmd = np.mean(K_XX) + np.mean(K_YY) - 2 * np.mean(K_XY)
        return max(0, mmd)
    
    def train_meta_model(self, transfer_results):
        """
        Train a meta-model to predict performance based on domain characteristics.
        
        Args:
            transfer_results: List of dicts with format:
                {'source_X': ..., 'source_y': ..., 
                 'target_X': ..., 'target_y': ...,
                 'model_type': 'lightgbm',
                 'actual_auc_pr': 0.XX}
        """
        print("Training transferability meta-model...")
        
        # Extract features and targets
        X_meta = []
        y_meta = []
        
        for result in transfer_results:
            # Compute domain similarity metrics
            metrics = self.compute_domain_similarity(
                result['source_X'], result['source_y'],
                result['target_X'], result['target_y']
            )
            
            # Add model type as feature
            model_features = {
                'is_tree_based': 1 if result['model_type'] in ['lightgbm', 'randomforest'] else 0,
                'is_neural': 1 if result['model_type'] in ['mlp'] else 0,
                'is_unsupervised': 1 if result['model_type'] in ['isolation_forest'] else 0
            }
            
            # Combine features
            features = {**metrics, **model_features}
            X_meta.append(list(features.values()))
            y_meta.append(result['actual_auc_pr'])
        
        X_meta = np.array(X_meta)
        y_meta = np.array(y_meta)
        
        # Train meta-model
        self.meta_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,  # Simple model to avoid overfitting
            random_state=42
        )
        self.meta_model.fit(X_meta, y_meta)
        
        # Store feature importance
        feature_names = list(metrics.keys()) + list(model_features.keys())
        self.feature_importance = dict(zip(feature_names, self.meta_model.feature_importances_))
        
        # Evaluate meta-model
        train_preds = self.meta_model.predict(X_meta)
        mse = np.mean((train_preds - y_meta) ** 2)
        mae = np.mean(np.abs(train_preds - y_meta))
        
        print(f"  Meta-model trained: MSE={mse:.4f}, MAE={mae:.4f}")
        print(f"  Most important factors for transfer success:")
        for factor, importance in sorted(self.feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)[:3]:
            print(f"    - {factor}: {importance:.3f}")
        
        return self
    
    def predict_performance_drop(self, source_X, source_y, target_X, target_y, model_type):
        """
        Predict expected AUC-PR on target domain WITHOUT training the model.
        
        Returns:
            dict with predicted metrics and confidence
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Call train_meta_model first.")
        
        # Compute domain similarity
        metrics = self.compute_domain_similarity(source_X, source_y, target_X, target_y)
        
        # Add model type features
        model_features = {
            'is_tree_based': 1 if model_type in ['lightgbm', 'randomforest'] else 0,
            'is_neural': 1 if model_type in ['mlp'] else 0,
            'is_unsupervised': 1 if model_type in ['isolation_forest'] else 0
        }
        
        # Combine features
        features = {**metrics, **model_features}
        X_meta = np.array(list(features.values())).reshape(1, -1)
        
        # Predict performance
        predicted_auc_pr = self.meta_model.predict(X_meta)[0]
        
        # Get prediction uncertainty (using forest variance)
        tree_predictions = [tree.predict(X_meta)[0] for tree in self.meta_model.estimators_]
        uncertainty = np.std(tree_predictions)
        
        # Determine risk level
        if metrics['overall_transferability'] < 0.3:
            risk_level = 'HIGH'
            recommendation = 'Avoid deployment - severe domain shift detected'
        elif metrics['overall_transferability'] < 0.6:
            risk_level = 'MEDIUM'
            recommendation = 'Proceed with caution - significant differences exist'
        else:
            risk_level = 'LOW'
            recommendation = 'Good transfer expected - domains are similar'
        
        return {
            'predicted_auc_pr': predicted_auc_pr,
            'uncertainty': uncertainty,
            'confidence_interval': (predicted_auc_pr - 2*uncertainty, 
                                   predicted_auc_pr + 2*uncertainty),
            'transferability_score': metrics['overall_transferability'],
            'risk_level': risk_level,
            'recommendation': recommendation,
            'detailed_metrics': metrics
        }
    
    def analyze_failure_modes(self, transfer_results):
        """
        Identify common patterns in transfer failures.
        
        Returns:
            Dictionary of failure patterns and recommendations
        """
        failures = []
        successes = []
        
        for result in transfer_results:
            metrics = self.compute_domain_similarity(
                result['source_X'], result['source_y'],
                result['target_X'], result['target_y']
            )
            
            if result['actual_auc_pr'] < 0.05:  # Severe failure
                failures.append(metrics)
            elif result['actual_auc_pr'] > 0.10:  # Relative success
                successes.append(metrics)
        
        if not failures:
            return {"error": "No failure cases found in data"}
        
        if not successes:
            # If no clear successes, use relative performance
            all_scores = [r['actual_auc_pr'] for r in transfer_results]
            median_score = np.median(all_scores)
            
            for result in transfer_results:
                metrics = self.compute_domain_similarity(
                    result['source_X'], result['source_y'],
                    result['target_X'], result['target_y']
                )
                if result['actual_auc_pr'] >= median_score:
                    successes.append(metrics)
        
        # Find distinguishing factors
        failure_patterns = {}
        
        for metric in failures[0].keys():
            if metric == 'overall_transferability':
                continue
                
            fail_values = [f[metric] for f in failures]
            success_values = [s[metric] for s in successes]
            
            # T-test to see if distributions differ
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(fail_values, success_values)
            
            if p_value < 0.05:  # Significant difference
                failure_patterns[metric] = {
                    'failure_mean': np.mean(fail_values),
                    'success_mean': np.mean(success_values),
                    'importance': abs(t_stat),
                    'threshold': (np.mean(fail_values) + np.mean(success_values)) / 2
                }
        
        # Sort by importance
        sorted_patterns = sorted(failure_patterns.items(), 
                                key=lambda x: x[1]['importance'], 
                                reverse=True)
        
        # Generate recommendations
        recommendations = []
        for pattern, stats in sorted_patterns[:3]:
            if stats['failure_mean'] < stats['success_mean']:
                recommendations.append(
                    f"Ensure {pattern} > {stats['threshold']:.2f} for successful transfer"
                )
            else:
                recommendations.append(
                    f"Beware when {pattern} > {stats['threshold']:.2f} - indicates likely failure"
                )
        
        return {
            'critical_factors': dict(sorted_patterns[:3]),
            'recommendations': recommendations,
            'failure_threshold': np.mean([f['overall_transferability'] for f in failures]),
            'success_threshold': np.mean([s['overall_transferability'] for s in successes])
        }


def create_transfer_results_from_experiments(bench):
    """
    Convert existing experiment results into format for meta-learning.
    This uses your already-computed results.
    """
    print("Preparing transfer results for meta-learning...")
    
    transfer_results = []
    
    # Define the transfers you've already tested
    transfers = [
        ('paysim', 'ieee_cis', 'lightgbm', 0.061),
        ('paysim', 'ieee_cis', 'mlp', 0.034),
        ('paysim', 'ieee_cis', 'isolation_forest', 0.056),
        ('paysim', 'european_cc', 'lightgbm', 0.014),
        ('paysim', 'european_cc', 'mlp', 0.002),
        ('ieee_cis', 'paysim', 'lightgbm', 0.045),
        ('ieee_cis', 'european_cc', 'lightgbm', 0.003),
        ('european_cc', 'ieee_cis', 'lightgbm', 0.041),
        ('european_cc', 'paysim', 'lightgbm', 0.131),
    ]
    
    for source, target, model_type, auc_pr in transfers:
        # Load datasets
        source_X, source_y = bench.load_processed_dataset(source)
        target_X, target_y = bench.load_processed_dataset(target)
        
        transfer_results.append({
            'source_X': source_X,
            'source_y': source_y,
            'target_X': target_X,
            'target_y': target_y,
            'model_type': model_type,
            'actual_auc_pr': auc_pr
        })
    
    return transfer_results


# Example usage
if __name__ == "__main__":
    from fraudbench import FraudBench
    
    # Initialize
    bench = FraudBench(data_dir='/home/paulj/niru/fraudbench/fraudbench_data')
    predictor = TransferabilityPredictor()
    
    # Create training data from your experiments
    transfer_results = create_transfer_results_from_experiments(bench)
    
    # Train the meta-model
    predictor.train_meta_model(transfer_results)
    
    # Now predict for a new transfer WITHOUT training
    print("\n" + "="*60)
    print("DEMONSTRATING TRANSFER PREDICTION")
    print("="*60)
    
    # Load data for prediction
    source_X, source_y = bench.load_processed_dataset('paysim')
    target_X, target_y = bench.load_processed_dataset('ieee_cis')
    
    # Predict performance BEFORE training any model
    prediction = predictor.predict_performance_drop(
        source_X, source_y, 
        target_X, target_y,
        model_type='randomforest'
    )
    
    print(f"\nPredicting RandomForest transfer from PaySim → IEEE-CIS:")
    print(f"  Predicted AUC-PR: {prediction['predicted_auc_pr']:.3f}")
    print(f"  Confidence interval: [{prediction['confidence_interval'][0]:.3f}, "
          f"{prediction['confidence_interval'][1]:.3f}]")
    print(f"  Transferability score: {prediction['transferability_score']:.3f}")
    print(f"  Risk level: {prediction['risk_level']}")
    print(f"  Recommendation: {prediction['recommendation']}")
    
    # Analyze failure modes
    print("\n" + "="*60)
    print("FAILURE MODE ANALYSIS")
    print("="*60)
    
    failure_analysis = predictor.analyze_failure_modes(transfer_results)
    print("\nCritical factors for transfer success:")
    for factor, stats in failure_analysis['critical_factors'].items():
        print(f"  {factor}: failure_avg={stats['failure_mean']:.3f}, "
              f"success_avg={stats['success_mean']:.3f}")
    
    print("\nRecommendations:")
    for rec in failure_analysis['recommendations']:
        print(f"  - {rec}")