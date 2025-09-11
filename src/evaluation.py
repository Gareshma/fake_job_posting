import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
import numpy as np


class ResultsLogger:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

    def save_results_to_file(self, trainer, filename=None):
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_results_{timestamp}.txt"

        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w') as f:
            f.write("FAKE JOB POSTING DETECTION - MODEL EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset shape: Train {trainer.X_train.shape}, Test {trainer.X_test.shape}\n")
            f.write("=" * 60 + "\n\n")

            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-" * 60 + "\n")
            f.write(f"{'Model':<20} {'Test Accuracy':<15} {'Test F1-Score':<15}\n")
            f.write("-" * 60 + "\n")

            for model_name, results in trainer.results.items():
                f.write(f"{model_name:<20} {results['test_accuracy']:<15.4f} {results['test_f1']:<15.4f}\n")

            best_accuracy_model = max(trainer.results.items(), key=lambda x: x[1]['test_accuracy'])
            best_f1_model = max(trainer.results.items(), key=lambda x: x[1]['test_f1'])

            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Best Accuracy: {best_accuracy_model[0]} ({best_accuracy_model[1]['test_accuracy']:.4f})\n")
            f.write(f"Best F1-Score: {best_f1_model[0]} ({best_f1_model[1]['test_f1']:.4f})\n")
            f.write("=" * 60 + "\n\n")

            for model_name, results in trainer.results.items():
                f.write(f"\nDETAILED RESULTS - {model_name.upper()}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Train Accuracy: {results['train_accuracy']:.4f}\n")
                f.write(f"Test Accuracy:  {results['test_accuracy']:.4f}\n")
                f.write(f"Train F1-Score: {results['train_f1']:.4f}\n")
                f.write(f"Test F1-Score:  {results['test_f1']:.4f}\n")
                f.write("\nClassification Report:\n")
                f.write(results['classification_report'])
                f.write("\n" + "-" * 50 + "\n")

        print(f"Results saved to: {filepath}")
        return filepath

    def create_results_dataframe(self, trainer):
        
        data = []
        for model_name, results in trainer.results.items():
            data.append({
                'Model': model_name,
                'Train_Accuracy': results['train_accuracy'],
                'Test_Accuracy': results['test_accuracy'],
                'Train_F1': results['train_f1'],
                'Test_F1': results['test_f1']
            })
        return pd.DataFrame(data)

    def save_results_csv(self, trainer, filename=None):
       
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_results_{timestamp}.csv"

        filepath = os.path.join(self.results_dir, filename)
        df = self.create_results_dataframe(trainer)
        df.to_csv(filepath, index=False)
        print(f"Results CSV saved to: {filepath}")

        # Generate bar chart, confusion matrix, feature importance, ROC
        self.plot_model_comparison(df)
        self.generate_additional_visuals(trainer)

        return filepath

    def plot_model_comparison(self, df, plot_filename='model_comparison_plot.png'):
       
        try:
            sns.set(style='whitegrid')
            plt.figure(figsize=(10, 6))
            bar_width = 0.35
            x = range(len(df))

            plt.bar(x, df['Test_Accuracy'], width=bar_width, label='Accuracy', color='skyblue')
            plt.bar([i + bar_width for i in x], df['Test_F1'], width=bar_width, label='F1 Score', color='salmon')

            plt.xlabel('Models')
            plt.ylabel('Score')
            plt.title('Model Comparison: Accuracy vs F1 Score')
            plt.xticks([i + bar_width / 2 for i in x], df['Model'], rotation=15)
            plt.ylim(0, 1)
            plt.legend()
            plt.tight_layout()

            output_path = os.path.join(self.results_dir, plot_filename)
            plt.savefig(output_path)
            plt.close()
            print(f"Model comparison chart saved to: {output_path}")
        except Exception as e:
            print(f"Failed to generate model comparison plot: {e}")

    def plot_confusion_matrix(self, model, X_test, y_test, model_name):
        
        try:
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues', values_format='d')
            plt.title(f'Confusion Matrix - {model_name}')
            output_path = os.path.join(self.results_dir, f'confusion_matrix_{model_name}.png')
            plt.savefig(output_path)
            plt.close()
            print(f"Confusion matrix saved for {model_name} at: {output_path}")
        except Exception as e:
            print(f"Could not generate confusion matrix for {model_name}: {e}")

    def plot_feature_importance(self, model, model_name='XGBoost'):
       
        try:
            import xgboost as xgb
            if isinstance(model, xgb.XGBClassifier):
                plt.figure(figsize=(10, 6))
                xgb.plot_importance(model, max_num_features=10, importance_type='weight')
                plt.title(f'Feature Importance - {model_name}')
                output_path = os.path.join(self.results_dir, f'feature_importance_{model_name}.png')
                plt.savefig(output_path)
                plt.close()
                print(f"Feature importance plot saved at: {output_path}")
            else:
                print(f"Skipped feature importance: {model_name} is not XGBoost.")
        except Exception as e:
            print(f"Failed to generate feature importance plot: {e}")

    def plot_roc_curve(self, model, X_test, y_test, model_name):
       
        try:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_score = model.decision_function(X_test)
            else:
                print(f"Cannot compute ROC for {model_name}: no probability or decision function.")
                return

            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc='lower right')

            output_path = os.path.join(self.results_dir, f'roc_curve_{model_name}.png')
            plt.savefig(output_path)
            plt.close()
            print(f"ROC curve saved for {model_name} at: {output_path}")
        except Exception as e:
            print(f"Could not generate ROC curve for {model_name}: {e}")

    def generate_additional_visuals(self, trainer):
       
        for model_name, model in trainer.model_objects.items():
            self.plot_confusion_matrix(model, trainer.X_test, trainer.y_test, model_name)
            self.plot_roc_curve(model, trainer.X_test, trainer.y_test, model_name)
            if model_name.lower() == 'xgboost':
                self.plot_feature_importance(model, model_name)
