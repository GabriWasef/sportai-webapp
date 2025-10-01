from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'pe-classifier'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PEClassifier:
    def __init__(self):
        self.models_data = None
        self.load_models()

    def load_models(self):
        """Carica i modelli dal file .pkl"""
        try:
            self.models_data = joblib.load('pe_models.pkl')
            logger.info("✅ Modelli caricati con successo!")
            logger.info(f"📊 Accuracy scores: {self.models_data['scores']}")
        except Exception as e:
            logger.error(f"❌ Errore caricamento modelli: {e}")
            raise

    def preprocess_data(self, data):
        """Preprocessa i dati per la predizione"""
        df = pd.DataFrame([data])

        # Encoding categoriche
        for col, encoder in self.models_data['encoders'].items():
            if col in df.columns:
                # Gestisce valori non visti
                if data[col] not in encoder.classes_:
                    df[col] = encoder.classes_[0]  # Default al primo valore
                else:
                    df[col] = encoder.transform([data[col]])[0]

        # Scaling numeriche
        numerical_cols = [col for col in df.columns if col not in self.models_data['encoders'].keys()]
        df[numerical_cols] = self.models_data['scaler'].transform(df[numerical_cols])

        # Ordina colonne come nel training
        df = df[self.models_data['feature_columns']]
        return df

    def predict(self, data, model_name='best'):
        """Fa la predizione"""
        try:
            # Preprocessa
            X_processed = self.preprocess_data(data)

            # Seleziona modello
            if model_name == 'best':
                # Trova il miglior modello
                best_model_name = max(self.models_data['scores'], key=self.models_data['scores'].get)
                model = self.models_data['models'][best_model_name]
            else:
                model = self.models_data['models'][model_name]
                best_model_name = model_name

            # Predizione
            prediction = model.predict(X_processed)[0]
            probabilities = model.predict_proba(X_processed)[0]

            # Decodifica risultato
            performance_classes = self.models_data['target_encoder'].classes_
            predicted_class = performance_classes[prediction]

            # Probabilità per classe
            prob_dict = {
                performance_classes[i]: float(probabilities[i])
                for i in range(len(performance_classes))
            }

            # Feature importance (solo per RF)
            feature_importance = []
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                for i, importance in enumerate(importances):
                    feature_importance.append({
                        'feature': self.models_data['feature_columns'][i],
                        'importance': float(importance)
                    })
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)

            return {
                'prediction': predicted_class,
                'confidence': float(max(probabilities)) * 100,
                'probabilities': prob_dict,
                'model_used': best_model_name,
                'feature_importance': feature_importance[:8]
            }

        except Exception as e:
            logger.error(f"Errore predizione: {e}")
            raise

    def get_all_predictions(self, data):
        """Ottiene predizioni da tutti i modelli"""
        results = {}
        for model_name in self.models_data['models'].keys():
            results[model_name] = self.predict(data, model_name)
        return results

    def get_model_comparison(self):
        """Confronto performance modelli"""
        return {
            'scores': self.models_data['scores'],
            'best_model': max(self.models_data['scores'], key=self.models_data['scores'].get)
        }

# Inizializza classificatore
classifier = PEClassifier()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Nessun dato fornito'}), 400

        model_name = data.pop('model', 'best')
        result = classifier.predict(data, model_name)

        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict-all', methods=['POST'])
def predict_all():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Nessun dato fornito'}), 400

        # Rimuovi model parameter se presente
        data.pop('model', None)
        results = classifier.get_all_predictions(data)

        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/models/comparison')
def models_comparison():
    try:
        comparison = classifier.get_model_comparison()
        return jsonify({'success': True, 'comparison': comparison})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/case-professor', methods=['POST'])
def case_professor():
    try:
        # Dati caso professore
        professor_data = {
            'Age': 16,
            'Gender': 'Other',
            'Grade_Level': '10th',
            'Strength_Score': 46.64215300709872,
            'Endurance_Score': 46.429237959680094,
            'Flexibility_Score': 50.36329185009137,
            'Speed_Agility_Score': 51.647608412365166,
            'BMI': 29.23722539893244,
            'Health_Fitness_Knowledge_Score': 83.46770775216494,
            'Skills_Score': 62.54601962323033,
            'Class_Participation_Level': 'Medium',
            'Attendance_Rate': 79.0904051034569,
            'Motivation_Level': 'Medium',
            'Overall_PE_Performance_Score': 71.98393517177486,
            'Improvement_Rate': 3.8062104305360576,
            'Final_Grade': 'A',
            'Previous_Semester_PE_Grade': 'B',
            'Hours_Physical_Activity_Per_Week': 6.141111485515476
        }

        results = classifier.get_all_predictions(professor_data)

        return jsonify({
            'success': True,
            'case_data': professor_data,
            'results': results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Avvio PE Performance Analyzer - Versione Moderna")
    print("🌐 Apri il browser su: http://localhost:5001")
    app.run(debug=True, port=5001, host='0.0.0.0')
