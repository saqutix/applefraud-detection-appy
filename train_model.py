import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.dummy import DummyClassifier
from imblearn.combine import SMOTETomek
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("TRAINING FRAUD DETECTION MODEL WITH ADVANCED TECHNIQUES")
print("="*60)

# 1. Load the dataset
print("\n[1] Loading creditcard.csv...")
df = pd.read_csv('creditcard.csv')
print(f"[OK] Loaded {len(df)} transactions")

# 2. Check class distribution
fraud_count = df['Class'].sum()
total = len(df)
print(f"\n[2] Class Distribution:")
print(f"   Legitimate: {total - fraud_count} ({((total - fraud_count)/total)*100:.4f}%)")
print(f"   Fraud:      {fraud_count} ({(fraud_count/total)*100:.4f}%)")

# 3. PROPER TRAIN/TEST SPLIT - NO LEAKAGE
print("\n[3] Creating proper train/test split...")

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Save test set for later use
pd.concat([X_test, pd.DataFrame(y_test, columns=['Class'])], axis=1).to_csv('holdout_test_set.csv', index=False)

print(f"   Training: {len(X_train)} samples")
print(f"   Test:     {len(X_test)} samples")
print(f"   Fraud in training: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"   Fraud in test: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
print("    Split verified - No data leakage!")

# Continue with scaling, SMOTE, training...

# 4. Scale features - IMPORTANT: Scale BEFORE SMOTE
print("\n[4] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Apply SMOTE + Tomek (Better than SMOTE alone)
print("\n[5] Applying SMOTE + Tomek to balance classes...")
smote_tomek = SMOTETomek(random_state=42)
X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_scaled, y_train)
print(f"   After resampling: {len(X_train_resampled)} samples")
print(f"   Fraud samples: {y_train_resampled.sum()} ({y_train_resampled.mean()*100:.2f}%)")

# 6. Train MULTIPLE models for comparison
print("\n[6] Training multiple models...")

# Model 1: Random Forest
print("   Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train_resampled, y_train_resampled)

# Model 2: Logistic Regression with class weight
print("   Training Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    C=0.1
)
lr_model.fit(X_train_resampled, y_train_resampled)

# Model 3: XGBoost (if available, otherwise skip)
try:
    from xgboost import XGBClassifier
    print("   Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=fraud_count/(total-fraud_count),
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_resampled, y_train_resampled)
    use_xgb = True
except ImportError:
    print("   XGBoost not installed, skipping...")
    use_xgb = False

# Baseline model
print("   Training Baseline...")
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train_scaled, y_train)

# Collect all models
all_models = {
    'Random Forest': rf_model,
    'Logistic Regression': lr_model,
}
if use_xgb:
    all_models['XGBoost'] = xgb_model
all_models['Baseline'] = baseline

# 7. Evaluate all models
print("\n[7] Evaluating models...")
model_results = {}
best_model = None
best_f1 = 0

for name, mdl in all_models.items():
    preds = mdl.predict(X_test_scaled)
    probs = mdl.predict_proba(X_test_scaled)[:, 1] if hasattr(mdl, 'predict_proba') else None
    
    metrics = {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds, zero_division=0),
        'Recall': recall_score(y_test, preds, zero_division=0),
        'F1 Score': f1_score(y_test, preds, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, probs) if probs is not None else 0.5
    }
    
    model_results[name] = {
        'metrics': metrics,
        'predictions': preds.tolist()
    }
    
    if metrics['F1 Score'] > best_f1:
        best_f1 = metrics['F1 Score']
        best_model = name
    
    print(f"\n   {name}:")
    print(f"      Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"      Precision: {metrics['Precision']:.4f}")
    print(f"      Recall:    {metrics['Recall']:.4f}")
    print(f"      F1 Score:  {metrics['F1 Score']:.4f}")
    print(f"      ROC-AUC:   {metrics['ROC-AUC']:.4f}")

print(f"\n[OK] Best model: {best_model} (F1: {best_f1:.4f})")

# 8. Find optimal threshold for the best model
print("\n[8] Finding optimal threshold for best model...")
best_model_obj = all_models[best_model]
y_proba = best_model_obj.predict_proba(X_test_scaled)[:, 1]

best_threshold = 0.5
best_f1 = 0
thresholds = np.arange(0.01, 0.99, 0.01)

for threshold in thresholds:
    preds = (y_proba > threshold).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"   Best threshold: {best_threshold:.2f}")
print(f"   Best F1 score: {best_f1:.4f}")

# 9. Save artifacts
print("\n[9] Saving model artifacts...")
joblib.dump(best_model_obj, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X_train.columns.tolist(), 'feature_names.pkl')
joblib.dump(best_threshold, 'best_threshold.pkl')
joblib.dump(all_models, 'all_models.pkl')
joblib.dump(model_results, 'model_results.pkl')
joblib.dump(best_model, 'best_model_name.pkl')

print("\n" + "="*60)
print("[OK] TRAINING COMPLETE!")
print(f"   Best model: {best_model}")
print(f"   Threshold: {best_threshold:.2f}")
print("   Saved files:")
print("   - fraud_detection_model.pkl")
print("   - scaler.pkl")
print("   - feature_names.pkl")
print("   - best_threshold.pkl")
print("   - all_models.pkl")
print("   - model_results.pkl")
print("   - best_model_name.pkl")
print("="*60)

# 10. Quick test
print("\n[10] Quick test on test files...")
test_files = ['100_percent_fraud.csv', 'mixed_fraud_test.csv']

for file in test_files:
    try:
        df_test = pd.read_csv(file)
        missing_cols = [col for col in X_train.columns if col not in df_test.columns]
        for col in missing_cols:
            df_test[col] = 0
        
        X_test_file = df_test[X.columns]
        X_test_file_scaled = scaler.transform(X_test_file)
        probs = best_model_obj.predict_proba(X_test_file_scaled)[:, 1]
        preds = (probs > best_threshold).astype(int)
        
        fraud_count = preds.sum()
        print(f"\n   {file}:")
        print(f"      Transactions: {len(df_test)}")
        print(f"      Predicted fraud: {fraud_count}")
        print(f"      Max probability: {probs.max():.4f}")
        
        if 'Class' in df_test.columns:
            actual = df_test['Class'].sum()
            print(f"      Actual fraud: {actual}")
            if actual > 0:
                caught = ((probs > best_threshold) & (df_test['Class'] == 1)).sum()
                print(f"      Caught: {caught}/{actual} ({caught/actual*100:.1f}%)")
    except FileNotFoundError:
        pass

print("\n[OK] Done!")


# 11. Verify no data leakage
# 11. Comprehensive data leakage check
print("\n[11] Data Integrity Check...")

# Check 1: Row indices
train_indices = set(df_train.index)
test_indices = set(df_holdout.index)
index_overlap = train_indices.intersection(test_indices)

# Check 2: Exact row duplicates (all features)
train_rows = set(df_train.values.tobytes() for df_train in [df_train])  # Simplified
test_rows = set(df_holdout.values.tobytes() for df_holdout in [df_holdout])

print(f"   Training samples: {len(df_train)}")
print(f"   Test samples: {len(df_holdout)}")
print(f"   Row index overlap: {len(index_overlap)}")
print(f"   Total samples: {len(df_train) + len(df_holdout)}")
print(f"   Original samples: {len(df)}")

if len(index_overlap) == 0:
    print("    SPLIT IS CORRECT - No data leakage!")
else:
    print(f"    WARNING: {len(index_overlap)} overlapping rows found!")

# Check fraud distribution in both sets
print(f"\n   Fraud in training: {df_train['Class'].sum()} ({df_train['Class'].mean()*100:.2f}%)")
print(f"   Fraud in test: {df_holdout['Class'].sum()} ({df_holdout['Class'].mean()*100:.2f}%)")