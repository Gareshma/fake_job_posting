import os

class Config:
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = True
    
    # Model paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, 'trained_models')
    
    XGBOOST_MODEL_PATH = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
    OCSVM_MODEL_PATH = os.path.join(MODELS_DIR, 'ocsvm_model.pkl')
    VECTORIZER_PATH = os.path.join(MODELS_DIR, 'vectorizer.pkl')
    LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')
    
    MODEL_METRICS = {
        'xgboost': {
            'f1_score': 0.94,
            'accuracy': 0.93
        },
        'ocsvm': {
            'f1_score': 0.88,
            'accuracy': 0.87
        }
    }
    
    # Swagger UI settings
    SWAGGER = {
        'title': 'Job Posting Classification API',
        'uiversion': 3,
        'description': 'API for detecting fake job postings using XGBoost and One-Class SVM',
        'version': '1.0.0'
    }