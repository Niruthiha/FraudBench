from fraudbench import FraudBench
from sklearn.ensemble import RandomForestClassifier

# Initialize
bench = FraudBench(data_dir='./fraudbench_data')
bench.prepare_datasets()

# Evaluate a model
model = RandomForestClassifier(n_estimators=100)
results = bench.evaluate(model, 'paysim', 'ieee_cis')
print(f"AUC-PR: {results['metrics']['auc_pr']['mean']:.3f}")
