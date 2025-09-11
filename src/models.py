import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.model_objects = {}  
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def split_data(self, X, y, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        print(f"Training class distribution: {np.bincount(self.y_train)}")
        print(f"Test class distribution: {np.bincount(self.y_test)}")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_logistic_regression(self, tune_hyperparameters=True):
        print("\n" + "=" * 50)
        print("Training Logistic Regression...")

        if tune_hyperparameters:
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'max_iter': [1000, 2000],
                'class_weight': [None, 'balanced']
            }
            lr = LogisticRegression(random_state=self.random_state)
            grid = GridSearchCV(lr, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
            grid.fit(self.X_train, self.y_train)
            self.models['logistic_regression'] = grid.best_estimator_
            print(f"Best parameters: {grid.best_params_}")
        else:
            self.models['logistic_regression'] = LogisticRegression(
                random_state=self.random_state, max_iter=1000, class_weight='balanced'
            )
            self.models['logistic_regression'].fit(self.X_train, self.y_train)

        self.model_objects['logistic_regression'] = self.models['logistic_regression']
        self._evaluate_model('logistic_regression')

    def train_naive_bayes(self):
        print("\n" + "=" * 50)
        print("Training Naive Bayes...")
        self.models['naive_bayes'] = GaussianNB()
        self.models['naive_bayes'].fit(self.X_train, self.y_train)
        self.model_objects['naive_bayes'] = self.models['naive_bayes']
        self._evaluate_model('naive_bayes')

    def train_random_forest(self, tune_hyperparameters=True):
        print("\n" + "=" * 50)
        print("Training Random Forest...")

        if tune_hyperparameters:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'class_weight': [None, 'balanced']
            }
            rf = RandomForestClassifier(random_state=self.random_state)
            grid = GridSearchCV(rf, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
            grid.fit(self.X_train, self.y_train)
            self.models['random_forest'] = grid.best_estimator_
            print(f"Best parameters: {grid.best_params_}")
        else:
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=200, random_state=self.random_state, class_weight='balanced'
            )
            self.models['random_forest'].fit(self.X_train, self.y_train)

        self.model_objects['random_forest'] = self.models['random_forest']
        self._evaluate_model('random_forest')

    def train_xgboost(self, tune_hyperparameters=True):
        print("\n" + "=" * 50)
        print("Training XGBoost...")

        neg = np.sum(self.y_train == 0)
        pos = np.sum(self.y_train == 1)
        scale_pos_weight = neg / pos

        if tune_hyperparameters:
            param_grid = {
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1],
                'n_estimators': [100, 200],
                'subsample': [0.8, 1.0],
                'scale_pos_weight': [1, scale_pos_weight]
            }
            xgb_model = xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss')
            grid = GridSearchCV(xgb_model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
            grid.fit(self.X_train, self.y_train)
            self.models['xgboost'] = grid.best_estimator_
            print(f"Best parameters: {grid.best_params_}")
        else:
            self.models['xgboost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1
            )
            self.models['xgboost'].fit(self.X_train, self.y_train)

        self.model_objects['xgboost'] = self.models['xgboost']
        self._evaluate_model('xgboost')

    def train_one_class_svm(self, tune_hyperparameters=True):
        print("\n" + "=" * 50)
        print("Training One-Class SVM (Anomaly Detection)...")
        X_normal = self.X_train[self.y_train == 0]

        if tune_hyperparameters:
            param_grid = {
                'kernel': ['rbf'],
                'gamma': ['scale', 0.01],
                'nu': [0.05, 0.1]
            }

            def one_class_scorer(estimator, X, y):
                preds = estimator.predict(X)
                preds = np.where(preds == -1, 1, 0)
                return f1_score(y, preds)

            ocsvm = OneClassSVM()
            grid = GridSearchCV(ocsvm, param_grid, cv=3, scoring=one_class_scorer, n_jobs=-1, verbose=1)
            grid.fit(X_normal, np.zeros(len(X_normal)))
            self.models['one_class_svm'] = grid.best_estimator_
            print(f"Best parameters: {grid.best_params_}")
        else:
            self.models['one_class_svm'] = OneClassSVM(kernel='rbf', gamma='scale', nu=0.1)
            self.models['one_class_svm'].fit(X_normal)

        self.model_objects['one_class_svm'] = self.models['one_class_svm']
        self._evaluate_one_class_svm()

    def _evaluate_model(self, name):
        model = self.models[name]
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)

        self.results[name] = {
            'train_accuracy': accuracy_score(self.y_train, y_train_pred),
            'test_accuracy': accuracy_score(self.y_test, y_test_pred),
            'train_f1': f1_score(self.y_train, y_train_pred),
            'test_f1': f1_score(self.y_test, y_test_pred),
            'classification_report': classification_report(self.y_test, y_test_pred)
        }

        print(f"Results for {name.upper()}:")
        print(f"  Train Accuracy: {self.results[name]['train_accuracy']:.4f}")
        print(f"  Test Accuracy: {self.results[name]['test_accuracy']:.4f}")
        print(f"  Train F1-Score: {self.results[name]['train_f1']:.4f}")
        print(f"  Test F1-Score: {self.results[name]['test_f1']:.4f}")

    def _evaluate_one_class_svm(self):
        model = self.models['one_class_svm']
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)

        y_pred_train = np.where(y_pred_train == -1, 1, 0)
        y_pred_test = np.where(y_pred_test == -1, 1, 0)

        self.results['one_class_svm'] = {
            'train_accuracy': accuracy_score(self.y_train, y_pred_train),
            'test_accuracy': accuracy_score(self.y_test, y_pred_test),
            'train_f1': f1_score(self.y_train, y_pred_train),
            'test_f1': f1_score(self.y_test, y_pred_test),
            'classification_report': classification_report(self.y_test, y_pred_test)
        }

        print("Results for ONE-CLASS SVM:")
        print(f"  Train Accuracy: {self.results['one_class_svm']['train_accuracy']:.4f}")
        print(f"  Test Accuracy: {self.results['one_class_svm']['test_accuracy']:.4f}")
        print(f"  Train F1-Score: {self.results['one_class_svm']['train_f1']:.4f}")
        print(f"  Test F1-Score: {self.results['one_class_svm']['test_f1']:.4f}")

    def train_all_models(self, tune_hyperparameters=False):
        print("Starting model training pipeline...")
        self.train_logistic_regression(tune_hyperparameters)
        self.train_naive_bayes()
        self.train_random_forest(tune_hyperparameters)
        self.train_xgboost(tune_hyperparameters)
        self.train_one_class_svm(tune_hyperparameters)
        print("\n" + "=" * 50)
        print("ALL MODELS TRAINED SUCCESSFULLY!")
        self.print_summary()

    def print_summary(self):
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"{'Model':<20} {'Test Accuracy':<15} {'Test F1-Score':<15}")
        print("-" * 50)

        for model_name, res in self.results.items():
            print(f"{model_name:<20} {res['test_accuracy']:<15.4f} {res['test_f1']:<15.4f}")

        best_acc = max(self.results.items(), key=lambda x: x[1]['test_accuracy'])
        best_f1 = max(self.results.items(), key=lambda x: x[1]['test_f1'])

        print("\n" + "=" * 60)
        print(f"Best Accuracy: {best_acc[0]} ({best_acc[1]['test_accuracy']:.4f})")
        print(f"Best F1-Score: {best_f1[0]} ({best_f1[1]['test_f1']:.4f})")
        print("=" * 60)
