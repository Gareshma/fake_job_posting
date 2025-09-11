import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.preprocessing import DataPreprocessor
from src.feature_extraction import FeatureExtractor
from src.models import ModelTrainer
from src.evaluation import ResultsLogger

def main():
    print("="*60)
    print("FAKE JOB POSTING DETECTION PIPELINE")
    print("="*60)
    
    # Configuration
    DATA_PATH = "data/fake_job_postings.csv"
    GLOVE_PATH = "embeddings/glove.6B.100d.txt"  
    TUNE_HYPERPARAMETERS = False  # Set to True for hyperparameter tuning (slower)
    
    try:
        # Step 1: Data Preprocessing
        print("\n1. DATA PREPROCESSING")
        print("-" * 30)
        
        preprocessor = DataPreprocessor()
        data = preprocessor.load_data(DATA_PATH)
        
        # Apply your existing preprocessing
        data = preprocessor.basic_preprocessing()
        data = preprocessor.process_location()
        data = preprocessor.clean_text_columns()
        
        # Analyze class distribution
        class_dist = preprocessor.get_class_distribution()
        
        # Prepare features and target
        X_categorical, X_numerical, X_text, y = preprocessor.prepare_features_target()
        
        # Step 2: Feature Extraction
        print("\n2. FEATURE EXTRACTION")
        print("-" * 30)
        
        feature_extractor = FeatureExtractor(embedding_dim=100)
        
        # Try to load GloVe embeddings
        glove_loaded = feature_extractor.load_glove_embeddings(GLOVE_PATH)
        
        if not glove_loaded:
            print("Using TF-IDF as fallback...")
        
        # Extract all features
        X, feature_names = feature_extractor.extract_all_features(
            X_categorical, X_numerical, X_text
        )
        
        # Step 3: Model Training
        print("\n3. MODEL TRAINING")
        print("-" * 30)
        
        trainer = ModelTrainer(random_state=42)
        
        # Split data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=0.2)
        
        # Train all models
        trainer.train_all_models(tune_hyperparameters=TUNE_HYPERPARAMETERS)
        
        # Step 4: Save Results
        print("\n4. SAVING RESULTS")
        print("-" * 30)
        
        logger = ResultsLogger()
        
        # Save detailed results to text file
        txt_file = logger.save_results_to_file(trainer, "model_results.txt")
        
        # Save summary to CSV
        csv_file = logger.save_results_csv(trainer, "model_results.csv")
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"Results saved to:")
        print(f"  • {txt_file}")
        print(f"  • {csv_file}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Please check your data path and dependencies.")
        return False
    
    return True

def download_glove_instructions():
    """Print instructions for downloading GloVe embeddings"""
    print("\nTo use GloVe embeddings (recommended):")
    print("1. Create 'embeddings' folder in your project directory")
    print("2. Download GloVe embeddings from: https://nlp.stanford.edu/projects/glove/")
    print("3. Download 'glove.6B.zip' (822 MB)")
    print("4. Extract 'glove.6B.100d.txt' to the 'embeddings' folder")
    print("\nAlternatively, the pipeline will use TF-IDF as fallback.")

if __name__ == "__main__":
    # Check if help requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print(__doc__)
        download_glove_instructions()
        sys.exit(0)
    
    # Check if GloVe download instructions requested
    if len(sys.argv) > 1 and sys.argv[1] in ['glove', 'download']:
        download_glove_instructions()
        sys.exit(0)
    
    # Run main pipeline
    success = main()
    sys.exit(0 if success else 1)