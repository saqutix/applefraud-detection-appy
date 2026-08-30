import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print(" PROPER MODEL TESTING - NO DATA LEAKAGE")
print("="*70)

# 1. Load the model and artifacts
print("\n[1] Loading model artifacts...")
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')
threshold = joblib.load('best_threshold.pkl')

print(f"    Model loaded: {type(model).__name__}")
print(f"    Threshold: {threshold:.3f}")

# 2. Load the HOLD OUT test set (THIS IS THE CRITICAL ONE)
print("\n[2] Loading holdout test set...")
try:
    df_holdout = pd.read_csv('holdout_test_set.csv')
    print(f"   Loaded {len(df_holdout)} holdout samples")
except FileNotFoundError:
    print("   holdout_test_set.csv not found!")
    print("   Please run train_model.py first to create it")
    exit()

# 3. Prepare holdout test data
X_holdout = df_holdout.drop('Class', axis=1)
y_holdout = df_holdout['Class']

# 4. Scale the holdout data
X_holdout_scaled = scaler.transform(X_holdout)

# 5. Make predictions on holdout data
print("\n[3] Evaluating on HOLDOUT TEST SET (UNSEEN DATA)...")
print("-" * 70)

# Get predictions
y_pred_holdout = model.predict(X_holdout_scaled)
y_proba_holdout = model.predict_proba(X_holdout_scaled)[:, 1]

# Apply threshold
y_pred_threshold = (y_proba_holdout > threshold).astype(int)

# Calculate metrics
metrics_holdout = {
    'Accuracy': accuracy_score(y_holdout, y_pred_threshold),
    'Precision': precision_score(y_holdout, y_pred_threshold),
    'Recall': recall_score(y_holdout, y_pred_threshold),
    'F1 Score': f1_score(y_holdout, y_pred_threshold),
    'ROC-AUC': roc_auc_score(y_holdout, y_proba_holdout)
}

print("\n    Performance on Holdout Set:")
print(f"   {'Metric':<15} {'Score':<10}")
print(f"   {'-'*25}")
print(f"   {'Accuracy':<15} {metrics_holdout['Accuracy']:.4f}")
print(f"   {'Precision':<15} {metrics_holdout['Precision']:.4f}")
print(f"   {'Recall':<15} {metrics_holdout['Recall']:.4f}")
print(f"   {'F1 Score':<15} {metrics_holdout['F1 Score']:.4f}")
print(f"   {'ROC-AUC':<15} {metrics_holdout['ROC-AUC']:.4f}")

# Classification report
print("\n    Classification Report:")
print(classification_report(y_holdout, y_pred_threshold, target_names=['Legit', 'Fraud']))

# Confusion Matrix
cm = confusion_matrix(y_holdout, y_pred_threshold)
tn, fp, fn, tp = cm.ravel()

print("\n   Confusion Matrix:")
print(f"   {'':<15} {'Predicted Legit':<20} {'Predicted Fraud':<20}")
print(f"   {'Actual Legit':<15} {tn:<20} {fp:<20}")
print(f"   {'Actual Fraud':<15} {fn:<20} {tp:<20}")

# Calculate key metrics from confusion matrix
if (tp + fp) > 0:
    precision = tp / (tp + fp)
else:
    precision = 0
if (tp + fn) > 0:
    recall = tp / (tp + fn)
else:
    recall = 0
if (tp + fp + tn + fn) > 0:
    accuracy = (tp + tn) / (tp + fp + tn + fn)

print(f"\n   Key Metrics:")
print(f"   True Negatives: {tn} (Correctly identified as legit)")
print(f"   False Positives: {fp} (Falsely flagged as fraud) ")
print(f"   False Negatives: {fn} (Missed frauds) ")
print(f"   True Positives: {tp} (Correctly caught frauds)")

print(f"\n   Performance Summary:")
print(f"   Caught {tp}/{tp+fn} frauds ({recall*100:.1f}% detection rate)")
print(f"   False alarm rate: {fp}/{fp+tn} ({fp/(fp+tn)*100 if (fp+tn)>0 else 0:.2f}% of legit flagged as fraud)")

# ============================================================
# 6. Test on custom files
# ============================================================
print("\n" + "="*70)
print("[4] Testing on CUSTOM test files...")
print("="*70)

test_files = ['test_100_percent_fraud.csv', 'test_mixed_fraud.csv', 'test_mixed_fraud_2.csv', 'test_small_sample.csv']

for file in test_files:
    try:
        print(f"\n {file}:")
        df_test = pd.read_csv(file)
        
        # Check if Class column exists
        has_class = 'Class' in df_test.columns
        
        # Prepare features
        X_test_file = df_test.drop('Class', axis=1) if has_class else df_test
        X_test_file_scaled = scaler.transform(X_test_file)
        
        # Predict
        probs = model.predict_proba(X_test_file_scaled)[:, 1]
        preds = (probs > threshold).astype(int)
        
        print(f"   Transactions: {len(df_test)}")
        print(f"   Predicted fraud: {preds.sum()}")
        print(f"   Fraud probability range: {probs.min():.4f} - {probs.max():.4f}")
        print(f"   Average probability: {probs.mean():.4f}")
        
        if has_class:
            actual = df_test['Class'].sum()
            print(f"   Actual fraud: {actual}")
            
            if actual > 0:
                caught = ((probs > threshold) & (df_test['Class'] == 1)).sum()
                legit_correct = ((probs <= threshold) & (df_test['Class'] == 0)).sum()
                legit_total = len(df_test) - actual
                
                print(f"   Caught: {caught}/{actual} ({caught/actual*100:.1f}%)")
                if legit_total > 0:
                    print(f"    Correctly identified as legit: {legit_correct}/{legit_total} ({legit_correct/legit_total*100:.1f}%)")
        
        # Show some examples of predictions
        print(f"\n   Sample predictions (first 10):")
        print(f"   {'Index':<6} {'Prob':<8} {'Predicted':<12} {'Actual':<8}")
        print(f"   {'-'*40}")
        for i in range(min(10, len(df_test))):
            actual_label = df_test['Class'].iloc[i] if has_class else '?'
            pred_label = 'FRAUD' if preds[i] == 1 else 'LEGIT'
            print(f"   {i:<6} {probs[i]:.4f}  {pred_label:<12} {actual_label:<8}")
            
    except FileNotFoundError:
        print(f"   File not found: {file}")
    except Exception as e:
        print(f"    Error: {e}")

# ============================================================
# 7. Check for data leakage (VERY IMPORTANT)
# ============================================================
print("\n" + "="*70)
print("[5] CHECKING FOR DATA LEAKAGE...")
print("="*70)

# Load training data (from the original split)
try:
    # We need to reconstruct training data from the holdout
    df_full = pd.read_csv('creditcard.csv')
    
    # Get the holdout indices (we can't know exactly, but we can check overlaps)
    holdout_times = set(df_holdout['Time'])
    train_times = set(df_full['Time']) - holdout_times
    
    # Check custom files for overlaps with training
    for file in ['test_100_percent_fraud.csv', 'test_mixed_fraud.csv', 'test_mixed_fraud_2.csv', 'test_small_sample.csv']:
        try:
            df_check = pd.read_csv(file)
            file_times = set(df_check['Time'])
            overlaps = file_times.intersection(train_times)
            
            print(f"\n   {file}:")
            print(f"      Total samples: {len(df_check)}")
            print(f"      Overlaps with training: {len(overlaps)}")
            if len(overlaps) > 0:
                print(f"       WARNING: {len(overlaps)} samples were in training data!")
            else:
                print(f"       No overlaps with training data")
        except FileNotFoundError:
            pass
            
except Exception as e:
    print(f"   Could not check leakage: {e}")

print("\n" + "="*70)
print(" TESTING COMPLETE!")
print("="*70)

# ============================================================
# 8. Visualize results (optional)
# ============================================================
try:
    print("\n[6] Generating confusion matrix visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Holdout confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_holdout, y_pred_threshold,
        display_labels=['Legit', 'Fraud'],
        ax=axes[0],
        cmap='Blues'
    )
    axes[0].set_title('Holdout Test Set\n(Unseen Data)')
    
    # Custom file confusion matrix (if available)
    if 'df_test' in locals() and 'Class' in df_test.columns:
        ConfusionMatrixDisplay.from_predictions(
            df_test['Class'], preds,
            display_labels=['Legit', 'Fraud'],
            ax=axes[1],
            cmap='Greens'
        )
        axes[1].set_title('Custom Test File')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("    Saved confusion matrices to 'confusion_matrices.png'")
except Exception as e:
    print(f"   Could not generate visualization: {e}")

print("\n" + "="*70)
print(" ALL TESTS COMPLETE!")
print("="*70)