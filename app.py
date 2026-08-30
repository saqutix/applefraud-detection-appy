from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import io
import base64
import json
import os
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)

app = Flask(__name__)
CORS(app)

# Load models and artifacts
model = None
scaler = None
feature_names = None
all_models = None
model_results = None
best_model_name = None
PREDICTION_THRESHOLD = 0.10  # <-- ADD THIS! Set to 0.10

def load_artifacts():
    """Load all saved models and artifacts"""
    global model, scaler, feature_names, all_models, model_results, best_model_name, PREDICTION_THRESHOLD
    
    try:
        # Try to load the best threshold from training
        if os.path.exists('best_threshold.pkl'):
            PREDICTION_THRESHOLD = joblib.load('best_threshold.pkl')
            print(f"[OK] Loaded threshold: {PREDICTION_THRESHOLD:.2f}")
        else:
            print(f"[WARNING] best_threshold.pkl not found. Using default: {PREDICTION_THRESHOLD}")

        if os.path.exists('best_model_name.pkl'):
            best_model_name = joblib.load('best_model_name.pkl')
            print(f"[OK] Best model name: {best_model_name}")
        
        if os.path.exists('fraud_detection_model.pkl'):
            model = joblib.load('fraud_detection_model.pkl')
            print("[OK] Main model loaded!")
        else:
            print("[WARNING] Model file not found. Please run train_model.py first.")
        
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
            print("[OK] Scaler loaded!")
        
        if os.path.exists('feature_names.pkl'):
            feature_names = joblib.load('feature_names.pkl')
            print(f"[OK] Feature names loaded: {len(feature_names)} features")
        else:
            feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            print("[WARNING] Using fallback feature names")
        
        # Load all models for comparison
        if os.path.exists('all_models.pkl'):
            all_models = joblib.load('all_models.pkl')
            print(f"[OK] All models loaded: {list(all_models.keys())}")
        
        if os.path.exists('model_results.pkl'):
            model_results = joblib.load('model_results.pkl')
            print(f"[OK] Model results loaded")
            if model_results:
                best_model_name = max(model_results.keys(), 
                                    key=lambda x: model_results[x]['metrics']['F1 Score'])
                print(f"[OK] Best model: {best_model_name}")
        
    except Exception as e:
        print(f"[ERROR] Error loading artifacts: {e}")
        traceback.print_exc()

# Load artifacts on startup
load_artifacts()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    """Get information about loaded models"""
    info = {
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_count': len(feature_names) if feature_names else 0,
        'models_available': list(all_models.keys()) if all_models else [],
        'best_model': best_model_name,
        'has_results': model_results is not None,
        'prediction_threshold': PREDICTION_THRESHOLD  # <-- ADD THIS
    }
    
    if model_results:
        comparison = {}
        for name, result in model_results.items():
            comparison[name] = result['metrics']
        info['model_comparison'] = comparison
    
    return jsonify(info)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({
            'error': 'Model not loaded. Please run train_model.py first.'
        }), 500
    
    try:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Check and handle missing columns
        missing_cols = [col for col in feature_names if col not in df.columns]
        if missing_cols:
            for col in missing_cols:
                df[col] = 0
                print(f"Added missing column: {col} with zeros")
        
        # Ensure correct column order
        df = df[feature_names]
        
        # Scale features
        X_scaled = scaler.transform(df)
        
        # ============================================
        # FORCE FRAUD DETECTION WITH 0.10 THRESHOLD
        # ============================================
        PREDICTION_THRESHOLD = 0.10  # FORCE THIS
        
        # Get probabilities
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # FORCE predictions based on threshold
        predictions = (probabilities > PREDICTION_THRESHOLD).astype(int)
        
        print(f"\n[FORCED] Threshold: {PREDICTION_THRESHOLD}")
        print(f"[FORCED] Fraud predictions: {predictions.sum()}")
        print(f"[FORCED] Max prob: {probabilities.max():.4f}")
        print(f"[FORCED] Sample probs: {probabilities[:5]}")
        # ============================================
        
        # Get predictions from all models if available
        all_model_predictions = {}
        if all_models:
            for name, mdl in all_models.items():
                probs = mdl.predict_proba(X_scaled)[:, 1]
                all_model_predictions[name] = {
                    'prediction': (probs > PREDICTION_THRESHOLD).astype(int).tolist(),
                    'probability': probs.tolist()
                }
        
        # Add results to dataframe
        df_result = df.copy()
        df_result['Prediction'] = predictions
        df_result['Fraud_Probability'] = probabilities
        df_result['Status'] = df_result['Prediction'].map({0: 'Legitimate', 1: 'FRAUD'})
        df_result['Risk_Level'] = pd.cut(
            probabilities, 
            bins=[-0.001, 0.05, 0.15, 0.30, 1.001],
            labels=['Very Low', 'Low', 'Medium', 'High']
        )
        
        # Statistics
        total = len(df_result)
        fraud_count = int(predictions.sum())
        fraud_rate = (fraud_count / total * 100) if total > 0 else 0
        
        print(f"\n[RESULTS] Total: {total}, Fraud: {fraud_count}, Rate: {fraud_rate:.2f}%")
        print(f"[RESULTS] First 10 predictions: {predictions[:10]}")
        print(f"[RESULTS] First 10 statuses: {df_result['Status'].head(10).tolist()}")
        
        # Calculate metrics (using predictions as "true" for demo)
        y_true = df_result['Prediction']
        y_pred = df_result['Prediction']
        y_proba = df_result['Fraud_Probability']
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5),
            'avg_precision': float(average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0)
        }
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Generate confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        cm_plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = [
                {'feature': feature_names[i], 'importance': float(importance[i])} 
                for i in range(len(feature_names))
            ]
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            feature_importance = feature_importance[:15]
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
            feature_importance = [
                {'feature': feature_names[i], 'importance': float(importance[i])} 
                for i in range(len(feature_names))
            ]
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            feature_importance = feature_importance[:15]
        
        # Prepare display data
        display_cols = ['Time', 'Amount', 'Fraud_Probability', 'Status', 'Risk_Level']
        display_data = df_result[display_cols].to_dict('records')
        
        results = {
            'total': total,
            'fraud_count': fraud_count,
            'legitimate_count': total - fraud_count,
            'fraud_rate': round(fraud_rate, 2),
            'predictions': display_data,
            'fraud_transactions': df_result[df_result['Prediction'] == 1][['Time', 'Amount', 'Fraud_Probability']].to_dict('records'),
            'risk_distribution': df_result['Risk_Level'].value_counts().to_dict(),
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_plot': cm_plot_base64,
            'feature_importance': feature_importance,
            'has_feature_importance': feature_importance is not None,
            'all_model_predictions': all_model_predictions,
            'model_comparison': {name: model_results[name]['metrics'] for name in model_results.keys()} if model_results else None,
            'best_model': best_model_name,
            'threshold_used': PREDICTION_THRESHOLD
        }
        
        # Save full results as CSV
        csv_buffer = io.StringIO()
        df_result.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        return jsonify({
            'success': True,
            'results': results,
            'csv_data': base64.b64encode(csv_data.encode()).decode('utf-8')
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/download_sample', methods=['GET'])
def download_sample():
    """Download a sample CSV file"""
    sample_data = {}
    sample_data['Time'] = [0, 1, 2, 3, 4]
    sample_data['Amount'] = [149.62, 2.69, 378.66, 123.50, 69.99]
    
    v_values = [
        [-1.35, -0.07, 2.53, 1.37, -0.33, 0.46, 0.23, 0.09, 0.36, -0.06],
        [1.19, 0.26, 0.16, 0.44, 0.06, -0.08, -0.07, 0.08, -0.25, -0.27],
        [-1.35, -1.34, 1.77, 0.37, -0.50, 1.80, 0.79, 0.24, -1.51, 0.24],
        [-0.96, -0.18, 1.79, -0.86, -0.01, 1.24, 0.23, 0.37, -1.38, 0.22],
        [-1.15, 0.87, 1.54, 0.40, -0.40, 0.09, 0.59, -0.27, 0.81, -0.04]
    ]
    
    df = pd.DataFrame(sample_data)
    for i in range(10):
        df[f'V{i+1}'] = [row[i] for row in v_values]
    for i in range(11, 29):
        df[f'V{i}'] = np.random.randn(5) * 0.5
    
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    return send_file(
        io.BytesIO(csv_buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='sample_transactions.csv'
    )

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get the status of all loaded components"""
    return jsonify({
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_names_loaded': feature_names is not None,
        'all_models_loaded': all_models is not None,
        'results_loaded': model_results is not None,
        'best_model': best_model_name,
        'feature_count': len(feature_names) if feature_names else 0,
        'prediction_threshold': PREDICTION_THRESHOLD  # <-- ADD THIS
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)