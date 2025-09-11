from flask import Flask, render_template, request, jsonify, send_from_directory
from flasgger import Swagger, swag_from
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os
import warnings
warnings.filterwarnings('ignore')

from config import Config
from sample_jobs import SAMPLE_JOBS, get_job_list, get_job_by_id

app = Flask(__name__)
app.config.from_object(Config)

# Initialize Swagger
swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Job Posting Classification API",
        "description": "API for detecting fake job postings using XGBoost and One-Class SVM",
        "version": "1.0.0"
    },
    "basePath": "/api",
    "schemes": ["http", "https"]
}
swagger = Swagger(app, template=swagger_template)

class JobClassifier:
    def __init__(self):
        self.xgboost_model = None
        self.ocsvm_model = None
        self.vectorizer = None
        self.label_encoder = None
        self.load_models()
    
    def load_models(self):
       
        try:
            os.makedirs(app.config['MODELS_DIR'], exist_ok=True)

            if not os.path.exists(app.config['XGBOOST_MODEL_PATH']):
                self.create_dummy_models()
            
            with open(app.config['XGBOOST_MODEL_PATH'], 'rb') as f:
                self.xgboost_model = pickle.load(f)
            with open(app.config['OCSVM_MODEL_PATH'], 'rb') as f:
                self.ocsvm_model = pickle.load(f)
            with open(app.config['VECTORIZER_PATH'], 'rb') as f:
                self.vectorizer = pickle.load(f)

            print("Models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            self.create_dummy_models()
    
    def create_dummy_models(self):
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import OneClassSVM

        print("Creating dummy models for demonstration...")

        texts = [job['description'] for job in SAMPLE_JOBS.values()]
        labels = [1 if job['label'] == 'real' else 0 for job in SAMPLE_JOBS.values()]
        
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X = self.vectorizer.fit_transform(texts)

        self.xgboost_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.xgboost_model.fit(X, labels)

        real_jobs_X = X[np.array(labels) == 1]
        self.ocsvm_model = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
        self.ocsvm_model.fit(real_jobs_X)

        with open(app.config['XGBOOST_MODEL_PATH'], 'wb') as f:
            pickle.dump(self.xgboost_model, f)
        with open(app.config['OCSVM_MODEL_PATH'], 'wb') as f:
            pickle.dump(self.ocsvm_model, f)
        with open(app.config['VECTORIZER_PATH'], 'wb') as f:
            pickle.dump(self.vectorizer, f)

        print("Dummy models created and saved!")

    def predict(self, job_text, model_type='both'):
       
        try:
            X = self.vectorizer.transform([job_text])
            results = {}

            if model_type in ['xgboost', 'both']:
                xgb_pred = self.xgboost_model.predict(X)[0]
                xgb_prob = self.xgboost_model.predict_proba(X)[0]
                results['xgboost'] = {
                    'prediction': 'real' if xgb_pred == 1 else 'fake',
                    'confidence': float(max(xgb_prob)),
                    'probabilities': {
                        'fake': float(xgb_prob[0]),
                        'real': float(xgb_prob[1])
                    }
                }

            if model_type in ['ocsvm', 'both']:
                ocsvm_pred = self.ocsvm_model.predict(X)[0]
                ocsvm_score = self.ocsvm_model.score_samples(X)[0]
                results['ocsvm'] = {
                    'prediction': 'real' if ocsvm_pred == 1 else 'fake',
                    'confidence': float(abs(ocsvm_score)),
                    'anomaly_score': float(ocsvm_score)
                }

            return results

        except Exception as e:
            return {'error': str(e)}

# Initialize model
classifier = JobClassifier()

@app.route('/')
def index():
    jobs = get_job_list()
    return render_template('index.html', jobs=jobs)

@app.route('/api/predict', methods=['POST'])
@swag_from({
    'tags': ['Prediction'],
    'description': 'Predict if a job posting is fake or real',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'job_id': {'type': 'integer'},
                    'job_text': {'type': 'string'},
                    'model_type': {'type': 'string', 'enum': ['xgboost', 'ocsvm', 'both'], 'default': 'both'}
                }
            }
        }
    ],
    'responses': {
        200: {
            'description': 'Prediction results',
            'schema': {
                'type': 'object',
                'properties': {
                    'job_info': {'type': 'object'},
                    'predictions': {'type': 'object'},
                    'model_metrics': {'type': 'object'}
                }
            }
        }
    }
})
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        job_text = None
        job_info = None

        if 'job_id' in data:
            job_id = int(data['job_id'])
            job_info = get_job_by_id(job_id)
            if job_info:
                job_text = f"{job_info['title']} {job_info['description']} {job_info['requirements']}"
            else:
                return jsonify({'error': 'Invalid job ID'}), 400
        elif 'job_text' in data:
            job_text = data['job_text']
        else:
            return jsonify({'error': 'Either job_id or job_text must be provided'}), 400

        model_type = data.get('model_type', 'both')
        predictions = classifier.predict(job_text, model_type)

        if 'error' in predictions:
            return jsonify({'error': predictions['error']}), 500

        response = {
            'job_info': job_info,
            'predictions': predictions,
            'model_metrics': app.config['MODEL_METRICS']
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/job/<int:job_id>')
@swag_from({
    'tags': ['Jobs'],
    'description': 'Get job details by ID',
    'parameters': [{'name': 'job_id', 'in': 'path', 'type': 'integer', 'required': True}],
    'responses': {
        200: {'description': 'Job details'},
        404: {'description': 'Not found'}
    }
})
def get_job(job_id):
    job = get_job_by_id(job_id)
    if job:
        return jsonify(job)
    else:
        return jsonify({'error': 'Job not found'}), 404

@app.route('/api/jobs')
@swag_from({
    'tags': ['Jobs'],
    'description': 'Get list of all sample jobs',
    'responses': {
        200: {'description': 'List of jobs'}
    }
})
def get_jobs():
    return jsonify(get_job_list())

# Serve evaluation plots from results folder
@app.route('/results/<path:filename>')
def serve_results_file(filename):
    return send_from_directory('results', filename)

# Evaluation visualization page
@app.route('/results')
def show_results():
    return render_template("result.html")

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
