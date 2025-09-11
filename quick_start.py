import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def quick_pipeline():
    print("QUICK START - FAKE JOB POSTING DETECTION")
    print("="*50)
    
    # Your existing preprocessing code
    print("Loading and preprocessing data...")
    data = pd.read_csv("data/fake_job_postings.csv")
    
    class_label = "fraudulent"
    nuniq = data.nunique().to_dict()
    categories = []
    for k, v in nuniq.items():
        if k == class_label:
            continue
        if v < 50:
            categories.append(k)
    
    drops = ["job_id"]
    columns = data.columns
    for col in columns:
        missing = data[col].isna().sum()
        if missing >= len(data) / 2:
            drops.append(col)
    
    data = data.drop(drops, axis=1)
    
    # Process location
    if 'location' in data.columns:
        location_split = data['location'].str.split(',', expand=True)
        location_split = location_split.reindex(columns=[0, 1, 2])
        location_split.columns = ['location_country', 'location_state', 'location_city']
        location_split = location_split.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
        data = data.join(location_split)
        data = data.drop(["location"], axis=1)
        categories.append("location_country")
        categories = list(set(categories))
    
    # Prepare features
    print("Preparing features...")
    
    # Combine text features
    text_cols = ['title', 'description', 'requirements', 'benefits', 'company_profile']
    existing_text_cols = [col for col in text_cols if col in data.columns]
    
    if existing_text_cols:
        data['combined_text'] = data[existing_text_cols].fillna('').apply(
            lambda row: ' '.join(row.values.astype(str)), axis=1
        )
    else:
        data['combined_text'] = ''
    
    # Extract text features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
    text_features = vectorizer.fit_transform(data['combined_text']).toarray()
    
    # Basic numerical features
    numerical_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64'] and col != 'fraudulent']
    if numerical_cols:
        scaler = StandardScaler()
        numerical_features = scaler.fit_transform(data[numerical_cols].fillna(0))
        X = np.concatenate([text_features, numerical_features], axis=1)
    else:
        X = text_features
    
    y = data['fraudulent'].values
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Class distribution: {Counter(y)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train models
    models = {}
    results = {}
    
    print("\nTraining models...")
    
    # 1. Logistic Regression
    print("- Logistic Regression")
    models['logistic_regression'] = LogisticRegression(random_state=42, max_iter=1000)
    models['logistic_regression'].fit(X_train, y_train)
    
    # 2. Naive Bayes
    print("- Naive Bayes")
    models['naive_bayes'] = GaussianNB()
    models['naive_bayes'].fit(X_train, y_train)
    
    # 3. Random Forest
    print("- Random Forest")
    models['random_forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
    models['random_forest'].fit(X_train, y_train)
    
    # 4. XGBoost
    print("- XGBoost")
    models['xgboost'] = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    models['xgboost'].fit(X_train, y_train)
    
    # 5. One-Class SVM
    print("- One-Class SVM")
    X_train_normal = X_train[y_train == 0]
    models['one_class_svm'] = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
    models['one_class_svm'].fit(X_train_normal)
    
    # Evaluate models
    print("\nEvaluating models...")
    
    for name, model in models.items():
        if name == 'one_class_svm':
            y_pred = model.predict(X_test)
            y_pred = np.where(y_pred == -1, 1, 0)  # Convert to fraud/legitimate
        else:
            y_pred = model.predict(X_test)
        
        accuracy  = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {'accuracy': accuracy, 'f1': f1}
        print(f"{name:20} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # Save results
    print("\nSaving results...")
    
    with open('results/quick_results.txt', 'w') as f:
        f.write("QUICK START RESULTS - FAKE JOB POSTING DETECTION\n")
        f.write("="*50 + "\n")
        f.write(f"Dataset shape: {X.shape}\n")
        f.write(f"Train/Test split: {X_train.shape[0]}/{X_test.shape[0]}\n")
        f.write(f"Class distribution: {Counter(y)}\n\n")
        
        f.write("MODEL RESULTS:\n")
        f.write("-"*50 + "\n")
        f.write(f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10}\n")
        f.write("-"*50 + "\n")
        
        for name, metrics in results.items():
            f.write(f"{name:<20} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f}\n")
        
        # Best models
        best_acc = max(results.items(), key=lambda x: x[1]['accuracy'])
        best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
        
        f.write(f"\nBest Accuracy: {best_acc[0]} ({best_acc[1]['accuracy']:.4f})\n")
        f.write(f"Best F1-Score: {best_f1[0]} ({best_f1[1]['f1']:.4f})\n")
    
    print("Results saved to: results/quick_results.txt")
    print("\nQuick pipeline completed successfully!")

if __name__ == "__main__":
    # Create results directory
    import os
    if not os.path.exists('results'):
        os.makedirs('results')
    
    quick_pipeline()