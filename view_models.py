import joblib
import pickle
import numpy as np
import pandas as pd
import os

print("="*70)
print("📊 COMPLETE MODEL INSPECTION")
print("="*70)

# Check all files
print("\n📁 Available Files:")
files = [
    'fraud_detection_model.pkl',
    'scaler.pkl', 
    'feature_names.pkl',
    'best_threshold.pkl',
    'best_model_name.pkl',
    'all_models.pkl',
    'model_results.pkl'
]

for f in files:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024
        print(f"  ✅ {f} ({size:.1f} KB)")
    else:
        print(f"  ❌ {f} (NOT FOUND)")

# ============================================
# 1. Main Model
# ============================================
print("\n" + "="*70)
print("[1] MAIN MODEL (fraud_detection_model.pkl)")
print("="*70)

try:
    model = joblib.load('fraud_detection_model.pkl')
    print(f"\n✅ Model loaded successfully!")
    print(f"  • Type: {type(model).__name__}")
    print(f"  • Class: {model.__class__.__name__}")
    
    # XGBoost specific
    if hasattr(model, 'get_params'):
        params = model.get_params()
        print(f"\n  📊 Model Parameters:")
        important_params = [
            'n_estimators', 'max_depth', 'learning_rate', 
            'subsample', 'colsample_bytree', 'scale_pos_weight'
        ]
        for p in important_params:
            if p in params:
                print(f"    • {p}: {params[p]}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print(f"\n  📈 Feature Importances:")
        feature_names = joblib.load('feature_names.pkl')
        importances = model.feature_importances_
        
        # Get top 10 features
        indices = np.argsort(importances)[-10:][::-1]
        for i in indices:
            print(f"    • {feature_names[i]}: {importances[i]:.4f}")
            
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================
# 2. Scaler
# ============================================
print("\n" + "="*70)
print("[2] SCALER (scaler.pkl)")
print("="*70)

try:
    scaler = joblib.load('scaler.pkl')
    print(f"\n✅ Scaler loaded successfully!")
    print(f"  • Type: {type(scaler).__name__}")
    
    if hasattr(scaler, 'center_'):
        print(f"  • Center shape: {scaler.center_.shape}")
        print(f"  • First 5 centers: {scaler.center_[:5]}")
        
    if hasattr(scaler, 'scale_'):
        print(f"  • Scale shape: {scaler.scale_.shape}")
        print(f"  • First 5 scales: {scaler.scale_[:5]}")
        
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================
# 3. Feature Names
# ============================================
print("\n" + "="*70)
print("[3] FEATURE NAMES (feature_names.pkl)")
print("="*70)

try:
    feature_names = joblib.load('feature_names.pkl')
    print(f"\n✅ Feature names loaded!")
    print(f"  • Total features: {len(feature_names)}")
    print(f"  • First 5: {feature_names[:5]}")
    print(f"  • Last 5: {feature_names[-5:]}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================
# 4. Best Threshold
# ============================================
print("\n" + "="*70)
print("[4] BEST THRESHOLD (best_threshold.pkl)")
print("="*70)

try:
    threshold = joblib.load('best_threshold.pkl')
    print(f"\n✅ Best threshold: {threshold:.3f}")
    print(f"  • This is the probability cutoff for fraud detection")
    print(f"  • Transactions with probability > {threshold:.3f} are flagged as fraud")
    
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================
# 5. Best Model Name
# ============================================
print("\n" + "="*70)
print("[5] BEST MODEL NAME (best_model_name.pkl)")
print("="*70)

try:
    best_model = joblib.load('best_model_name.pkl')
    print(f"\n✅ Best model: {best_model}")
    
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================
# 6. All Models Comparison
# ============================================
print("\n" + "="*70)
print("[6] ALL MODELS COMPARISON (all_models.pkl)")
print("="*70)

try:
    all_models = joblib.load('all_models.pkl')
    print(f"\n✅ All models loaded!")
    print(f"  • Models available: {list(all_models.keys())}")
    
    for name, mdl in all_models.items():
        print(f"\n  📊 {name}:")
        print(f"    • Type: {type(mdl).__name__}")
        
        if hasattr(mdl, 'n_estimators'):
            print(f"    • n_estimators: {mdl.n_estimators}")
        if hasattr(mdl, 'max_depth'):
            print(f"    • max_depth: {mdl.max_depth}")
        if hasattr(mdl, 'classes_'):
            print(f"    • Classes: {mdl.classes_}")
            
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================
# 7. Model Results
# ============================================
print("\n" + "="*70)
print("[7] MODEL RESULTS (model_results.pkl)")
print("="*70)

try:
    results = joblib.load('model_results.pkl')
    print(f"\n✅ Results loaded!")
    print(f"  • Models in results: {list(results.keys())}")
    
    # Create comparison table
    print("\n  📊 Performance Comparison:")
    print("  " + "-"*70)
    print(f"  {'Model':<22} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'ROC-AUC':<10}")
    print("  " + "-"*70)
    
    for name, data in results.items():
        if 'metrics' in data:
            m = data['metrics']
            print(f"  {name:<22} {m['Accuracy']:.4f}    {m['Precision']:.4f}    {m['Recall']:.4f}    {m['F1 Score']:.4f}    {m['ROC-AUC']:.4f}")
    
    print("  " + "-"*70)
    
    # Find best model from results
    best = max(results.items(), key=lambda x: x[1]['metrics']['F1 Score'])
    print(f"\n  🏆 Best model: {best[0]} (F1: {best[1]['metrics']['F1 Score']:.4f})")
            
except Exception as e:
    print(f"❌ Error: {e}")

# ============================================
# 8. Test Data Files
# ============================================
print("\n" + "="*70)
print("[8] TEST DATA FILES")
print("="*70)

test_files = [
    'real_100_percent_fraud.csv',
    'real_mixed_fraud_test.csv',
    '100_percent_fraud.csv',
    'mixed_fraud_test.csv'
]

for f in test_files:
    if os.path.exists(f):
        try:
            df = pd.read_csv(f)
            print(f"\n  📄 {f}:")
            print(f"    • Transactions: {len(df)}")
            if 'Class' in df.columns:
                fraud = df['Class'].sum()
                legit = len(df) - fraud
                print(f"    • Fraud: {fraud} ({fraud/len(df)*100:.1f}%)")
                print(f"    • Legit: {legit} ({legit/len(df)*100:.1f}%)")
            if 'Amount' in df.columns:
                print(f"    • Amount range: ${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}")
        except Exception as e:
            print(f"  ❌ Error reading {f}: {e}")

print("\n" + "="*70)
print("✅ INSPECTION COMPLETE!")
print("="*70)