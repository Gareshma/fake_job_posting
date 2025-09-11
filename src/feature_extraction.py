import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import os
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
def ensure_nltk_dependencies():
    
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    nltk_downloads = [
        'punkt_tab',
        'punkt', 
        'stopwords', 
        'wordnet', 
        'omw-1.4'
    ]
    
    for resource in nltk_downloads:
        try:
            nltk.download(resource, quiet=True)
            print(f"Downloaded {resource}")
        except Exception as e:
            print(f"Could not download {resource}: {e}")

# Call the function to ensure dependencies
ensure_nltk_dependencies()

class ImprovedFeatureExtractor:
    def __init__(self, embedding_dim=100, use_embeddings=True, max_tfidf_features=5000):
        self.embedding_dim = embedding_dim
        self.use_embeddings = use_embeddings
        self.max_tfidf_features = max_tfidf_features
        self.glove_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.one_hot_encoders = {}
        self.tfidf_vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize stop words with fallback
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("NLTK stopwords not available, using basic stopwords")
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        self.is_fitted = False
        
    def _ensure_dataframe(self, data, column_name=None):
        
        if isinstance(data, pd.Series):
            if column_name is None:
                column_name = data.name if data.name is not None else 'column_0'
            return pd.DataFrame({column_name: data})
        elif isinstance(data, pd.DataFrame):
            return data
        else:
            raise ValueError("Data must be a pandas Series or DataFrame")
    
    def validate_and_separate_data(self, data):
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        
        numerical_cols = []
        categorical_cols = []
        text_cols = []
        
        for col in data.columns:
            # Check if column contains text (long strings)
            sample_values = data[col].dropna().head(100)
            if len(sample_values) == 0:
                continue
                
            # Check for text columns (average string length > 50 or contains spaces)
            if sample_values.dtype == 'object':
                avg_length = sample_values.astype(str).str.len().mean()
                has_spaces = sample_values.astype(str).str.contains(' ').any()
                
                if avg_length > 50 or has_spaces:
                    text_cols.append(col)
                else:
                    # Check if it can be converted to numeric
                    try:
                        pd.to_numeric(sample_values, errors='raise')
                        numerical_cols.append(col)
                    except (ValueError, TypeError):
                        categorical_cols.append(col)
            else:
                # Try to determine if it's numerical
                try:
                    pd.to_numeric(sample_values, errors='raise')
                    numerical_cols.append(col)
                except (ValueError, TypeError):
                    categorical_cols.append(col)
        
        print(f"Data separation results:")
        print(f"  - Numerical columns ({len(numerical_cols)}): {numerical_cols[:5]}{'...' if len(numerical_cols) > 5 else ''}")
        print(f"  - Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
        print(f"  - Text columns ({len(text_cols)}): {text_cols[:3]}{'...' if len(text_cols) > 3 else ''}")
        
        # Return DataFrames, ensuring they maintain the original index
        return {
            'numerical': data[numerical_cols] if numerical_cols else pd.DataFrame(index=data.index),
            'categorical': data[categorical_cols] if categorical_cols else pd.DataFrame(index=data.index),
            'text': data[text_cols] if text_cols else pd.DataFrame(index=data.index)
        }
    
    def find_local_glove_embeddings(self, embedding_dim=None):
       
        if embedding_dim is None:
            embedding_dim = self.embedding_dim
            
        print(f"Looking for local GloVe embeddings ({embedding_dim}d)...")
        
        # Define possible local paths
        possible_paths = [
            f"embeddings/glove.6B.{embedding_dim}d.txt",
            f"./embeddings/glove.6B.{embedding_dim}d.txt",
            f"glove.6B.{embedding_dim}d.txt",
            f"./glove.6B.{embedding_dim}d.txt",
            f"data/glove.6B.{embedding_dim}d.txt",
            f"./data/glove.6B.{embedding_dim}d.txt",
            f"data/embeddings/glove.6B.{embedding_dim}d.txt",
            f"./data/embeddings/glove.6B.{embedding_dim}d.txt",
            f"../embeddings/glove.6B.{embedding_dim}d.txt",
            f"../data/glove.6B.{embedding_dim}d.txt"
        ]
        
        # Check if any of the files exist
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Found GloVe embeddings at: {path}")
                return path
        
        print(f"Local GloVe embeddings ({embedding_dim}d) not found in standard locations.")
        print("Will use TF-IDF as fallback...")
        return None
    
    def load_glove_embeddings(self, glove_path=None):
        
        print("Loading GloVe embeddings...")
        
        # Try to load from provided path first
        if glove_path and os.path.exists(glove_path):
            print(f"Using provided path: {glove_path}")
            self.glove_model = self._load_glove_from_file(glove_path)
            return self.glove_model is not None
        
        # Try to find local embeddings
        if self.use_embeddings:
            local_path = self.find_local_glove_embeddings(self.embedding_dim)
            if local_path:
                self.glove_model = self._load_glove_from_file(local_path)
                return self.glove_model is not None
        
        print("GloVe embeddings not available. Using TF-IDF as fallback...")
        return False
    
    def _load_glove_from_file(self, filepath):
        
        embeddings = {}
        total_lines = 0
        skipped_lines = 0
        
        try:
            # First pass to count lines for progress bar
            with open(filepath, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            
            # Second pass to load embeddings
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in tqdm(f, total=total_lines, desc="Loading GloVe"):
                    line = line.strip()
                    if not line:
                        continue
                        
                    values = line.split()
                    if len(values) != self.embedding_dim + 1:
                        skipped_lines += 1
                        continue
                    
                    word = values[0]
                    try:
                        vector = np.asarray(values[1:], dtype='float32')
                        if len(vector) == self.embedding_dim:
                            embeddings[word] = vector
                    except ValueError:
                        skipped_lines += 1
                        continue
            
            print(f"Loaded {len(embeddings)} word embeddings")
            if skipped_lines > 0:
                print(f"Skipped {skipped_lines} malformed lines")
            return embeddings
            
        except Exception as e:
            print(f"Error loading GloVe file: {e}")
            return None
    
    def preprocess_text(self, text):
       
        if pd.isna(text) or text == '' or text is None:
            return []
        
        # Convert to string and clean
        text = str(text).lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize with fallback
        try:
            tokens = word_tokenize(text)
        except (LookupError, OSError):
            # Fallback to simple split if NLTK punkt is not available
            tokens = text.split()
        
        # Remove stopwords, lemmatize, and filter
        processed_tokens = []
        for token in tokens:
            if token.isalpha() and len(token) > 1 and token not in self.stop_words:
                try:
                    lemmatized = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
                except:
                    # Fallback to original token if lemmatization fails
                    processed_tokens.append(token)
        
        return processed_tokens
    
    def get_text_embedding(self, text):
       
        tokens = self.preprocess_text(text)
        
        if not tokens:
            return np.zeros(self.embedding_dim)
        
        embeddings = []
        for token in tokens:
            if token in self.glove_model:
                embeddings.append(self.glove_model[token])
        
        if not embeddings:
            return np.zeros(self.embedding_dim)
        
        # Average the embeddings
        return np.mean(embeddings, axis=0)
    
    def extract_text_features(self, X_text, fit=True):
        
        # Handle empty input
        if X_text is None:
            return np.array([]).reshape(0, 0)
        
        # Ensure DataFrame format
        X_text = self._ensure_dataframe(X_text, 'text_column')
        
        if X_text.empty:
            return np.array([]).reshape(0, 0)
            
        print("Extracting text features...")
        
        # Combine all text columns into one if multiple columns exist
        if len(X_text.columns) > 1:
            combined_text = X_text.apply(lambda row: ' '.join([str(val) for val in row if pd.notna(val)]), axis=1)
        else:
            combined_text = X_text.iloc[:, 0]
        
        if self.glove_model is None:
            print("Using TF-IDF for text features...")
            return self._extract_tfidf_features(combined_text, fit=fit)
        
        print("Using GloVe embeddings for text features...")
        embeddings = []
        for text in tqdm(combined_text, desc="Processing text"):
            embedding = self.get_text_embedding(text)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _extract_tfidf_features(self, X_text, fit=True):
       
        print("Extracting TF-IDF features...")
        
        # Preprocess text for TF-IDF
        processed_texts = []
        for text in X_text:
            if pd.isna(text) or text == '' or text is None:
                processed_texts.append('')
            else:
                tokens = self.preprocess_text(text)
                processed_texts.append(' '.join(tokens))
        
        if fit or self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.max_tfidf_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(processed_texts).toarray()
        else:
            tfidf_features = self.tfidf_vectorizer.transform(processed_texts).toarray()
        
        print(f"TF-IDF features shape: {tfidf_features.shape}")
        return tfidf_features
    
    def extract_categorical_features(self, X_categorical, fit=True):
        
        # Handle empty input
        if X_categorical is None:
            return np.array([]).reshape(0, 0)
        
        # Ensure DataFrame format
        X_categorical = self._ensure_dataframe(X_categorical, 'categorical_column')
        
        if X_categorical.empty:
            return np.array([]).reshape(len(X_categorical), 0)
        
        print("Processing categorical features...")
        encoded_features = []
        
        for column in X_categorical.columns:
            # Fill NaN values and convert to string
            X_cat_col = X_categorical[column].fillna('unknown').astype(str)
            
            # Determine encoding strategy
            unique_values = X_cat_col.nunique()
            
            if unique_values > 10:  # Label encoding for high cardinality
                if fit or column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                    # Add 'unknown' to handle unseen categories
                    unique_vals = list(X_cat_col.unique()) + ['unknown']
                    self.label_encoders[column].fit(unique_vals)
                
                # Handle unseen categories
                encoded = []
                for val in X_cat_col:
                    try:
                        encoded.append(self.label_encoders[column].transform([val])[0])
                    except ValueError:
                        # Map unseen values to 'unknown'
                        encoded.append(self.label_encoders[column].transform(['unknown'])[0])
                
                encoded_features.append(np.array(encoded).reshape(-1, 1))
                
            else:  # One-hot encoding for low cardinality
                if fit or column not in self.one_hot_encoders:
                    self.one_hot_encoders[column] = OneHotEncoder(
                        sparse_output=False, 
                        handle_unknown='ignore',
                        drop='if_binary' if unique_values == 2 else None
                    )
                    encoded = self.one_hot_encoders[column].fit_transform(X_cat_col.values.reshape(-1, 1))
                else:
                    encoded = self.one_hot_encoders[column].transform(X_cat_col.values.reshape(-1, 1))
                
                encoded_features.append(encoded)
        
        if encoded_features:
            result = np.concatenate(encoded_features, axis=1)
            print(f"Categorical features shape: {result.shape}")
            return result
        else:
            return np.array([]).reshape(len(X_categorical), 0)
    
    def extract_numerical_features(self, X_numerical, fit=True):
       
        # Handle empty input
        if X_numerical is None:
            return np.array([]).reshape(0, 0)
        
        # Ensure DataFrame format
        X_numerical = self._ensure_dataframe(X_numerical, 'numerical_column')
        
        if X_numerical.empty:
            return np.array([]).reshape(len(X_numerical), 0)
        
        print("Processing numerical features...")
        
        # Convert columns to numeric, coercing errors to NaN
        X_numerical_clean = X_numerical.copy()
        for col in X_numerical_clean.columns:
            X_numerical_clean[col] = pd.to_numeric(X_numerical_clean[col], errors='coerce')
        
        # Fill NaN values with median (or 0 if all values are NaN)
        for col in X_numerical_clean.columns:
            median_val = X_numerical_clean[col].median()
            if pd.isna(median_val):
                median_val = 0
            X_numerical_clean[col] = X_numerical_clean[col].fillna(median_val)
        
        # Scale features
        if fit:
            scaled_features = self.scaler.fit_transform(X_numerical_clean)
        else:
            scaled_features = self.scaler.transform(X_numerical_clean)
        
        print(f"Numerical features shape: {scaled_features.shape}")
        return scaled_features
    
    def fit(self, data, glove_path=None):
       
        print("Fitting feature extractor...")
        
        # Ensure DataFrame format
        data = self._ensure_dataframe(data, 'data_column')
        
        # Separate data types
        separated_data = self.validate_and_separate_data(data)
        
        # Load embeddings if using them
        if self.use_embeddings:
            self.load_glove_embeddings(glove_path)
        
        # Fit all feature extractors
        if not separated_data['text'].empty:
            self.extract_text_features(separated_data['text'], fit=True)
        if not separated_data['categorical'].empty:
            self.extract_categorical_features(separated_data['categorical'], fit=True)
        if not separated_data['numerical'].empty:
            self.extract_numerical_features(separated_data['numerical'], fit=True)
        
        self.is_fitted = True
        print("Feature extractor fitted successfully")
        return self
    
    def transform(self, data):
        
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform. Call fit() first.")
        
        # Ensure DataFrame format
        data = self._ensure_dataframe(data, 'data_column')
        
        # Separate data types
        separated_data = self.validate_and_separate_data(data)
        
        return self._extract_all_features(separated_data, fit_mode=False)
    
    def fit_transform(self, X_categorical=None, X_numerical=None, X_text=None, glove_path=None, data=None):
       
        # New interface - single DataFrame
        if data is not None:
            self.fit(data, glove_path)
            return self.transform(data)
        
        # Old interface - separate DataFrames/Series
        if X_categorical is not None or X_numerical is not None or X_text is not None:
            # Ensure DataFrame format for all inputs
            if X_categorical is not None:
                X_categorical = self._ensure_dataframe(X_categorical, 'categorical')
            else:
                X_categorical = pd.DataFrame()
                
            if X_numerical is not None:
                X_numerical = self._ensure_dataframe(X_numerical, 'numerical')
            else:
                X_numerical = pd.DataFrame()
                
            if X_text is not None:
                X_text = self._ensure_dataframe(X_text, 'text')
            else:
                X_text = pd.DataFrame()
            
            return self.extract_all_features(X_categorical, X_numerical, X_text, glove_path, fit_mode=True)
        
        raise ValueError("Must provide either 'data' parameter or at least one of X_categorical, X_numerical, X_text")
    
    def extract_all_features(self, X_categorical, X_numerical, X_text, glove_path=None, fit_mode=True):
        
        print("Extracting all features...")
        
        # Ensure DataFrame format for all inputs
        X_categorical = self._ensure_dataframe(X_categorical, 'categorical') if X_categorical is not None else pd.DataFrame()
        X_numerical = self._ensure_dataframe(X_numerical, 'numerical') if X_numerical is not None else pd.DataFrame()
        X_text = self._ensure_dataframe(X_text, 'text') if X_text is not None else pd.DataFrame()
        
        # Load embeddings if using them and in fit mode
        if self.use_embeddings and fit_mode:
            self.load_glove_embeddings(glove_path)
        
        # Extract individual feature types
        text_features = self.extract_text_features(X_text, fit=fit_mode)
        categorical_features = self.extract_categorical_features(X_categorical, fit=fit_mode)
        numerical_features = self.extract_numerical_features(X_numerical, fit=fit_mode)
        
        # Determine the number of samples
        n_samples = 0
        if text_features.size > 0:
            n_samples = text_features.shape[0]
        elif categorical_features.size > 0:
            n_samples = categorical_features.shape[0]
        elif numerical_features.size > 0:
            n_samples = numerical_features.shape[0]
        elif not X_text.empty:
            n_samples = len(X_text)
        elif not X_categorical.empty:
            n_samples = len(X_categorical)
        elif not X_numerical.empty:
            n_samples = len(X_numerical)
        
        # Combine features
        feature_arrays = []
        feature_names = []
        
        if text_features.size > 0 and text_features.shape[0] > 0:
            feature_arrays.append(text_features)
            if self.glove_model is not None:
                feature_names.extend([f'text_emb_{i}' for i in range(text_features.shape[1])])
            else:
                feature_names.extend([f'tfidf_{i}' for i in range(text_features.shape[1])])
        
        if categorical_features.size > 0 and categorical_features.shape[0] > 0:
            feature_arrays.append(categorical_features)
            feature_names.extend([f'cat_{i}' for i in range(categorical_features.shape[1])])
        
        if numerical_features.size > 0 and numerical_features.shape[0] > 0:
            feature_arrays.append(numerical_features)
            feature_names.extend([f'num_{i}' for i in range(numerical_features.shape[1])])
        
        if feature_arrays:
            combined_features = np.concatenate(feature_arrays, axis=1)
        else:
            combined_features = np.array([]).reshape(n_samples, 0)
        
        print(f"Final feature matrix shape: {combined_features.shape}")
        print(f"Feature breakdown:")
        print(f"  - Text features: {text_features.shape[1] if text_features.size > 0 else 0}")
        print(f"  - Categorical features: {categorical_features.shape[1] if categorical_features.size > 0 else 0}")
        print(f"  - Numerical features: {numerical_features.shape[1] if numerical_features.size > 0 else 0}")
        
        return combined_features, feature_names
    
    def _extract_all_features(self, separated_data, fit_mode=True):
        
        print("Extracting all features...")
        
        # Extract individual feature types
        text_features = self.extract_text_features(separated_data['text'], fit=fit_mode)
        categorical_features = self.extract_categorical_features(separated_data['categorical'], fit=fit_mode)
        numerical_features = self.extract_numerical_features(separated_data['numerical'], fit=fit_mode)
        
        # Determine the number of samples
        n_samples = 0
        if text_features.size > 0:
            n_samples = text_features.shape[0]
        elif categorical_features.size > 0:
            n_samples = categorical_features.shape[0]
        elif numerical_features.size > 0:
            n_samples = numerical_features.shape[0]
        elif not separated_data['text'].empty:
            n_samples = len(separated_data['text'])
        elif not separated_data['categorical'].empty:
            n_samples = len(separated_data['categorical'])
        elif not separated_data['numerical'].empty:
            n_samples = len(separated_data['numerical'])
        
        # Combine features
        feature_arrays = []
        feature_names = []
        
        if text_features.size > 0 and text_features.shape[0] > 0:
            feature_arrays.append(text_features)
            if self.glove_model is not None:
                feature_names.extend([f'text_emb_{i}' for i in range(text_features.shape[1])])
            else:
                feature_names.extend([f'tfidf_{i}' for i in range(text_features.shape[1])])
        
        if categorical_features.size > 0 and categorical_features.shape[0] > 0:
            feature_arrays.append(categorical_features)
            feature_names.extend([f'cat_{i}' for i in range(categorical_features.shape[1])])
        
        if numerical_features.size > 0 and numerical_features.shape[0] > 0:
            feature_arrays.append(numerical_features)
            feature_names.extend([f'num_{i}' for i in range(numerical_features.shape[1])])
        
        if feature_arrays:
            combined_features = np.concatenate(feature_arrays, axis=1)
        else:
            combined_features = np.array([]).reshape(n_samples, 0)
        
        print(f"Final feature matrix shape: {combined_features.shape}")
        print(f"Feature breakdown:")
        print(f"  - Text features: {text_features.shape[1] if text_features.size > 0 else 0}")
        print(f"  - Categorical features: {categorical_features.shape[1] if categorical_features.size > 0 else 0}")
        print(f"  - Numerical features: {numerical_features.shape[1] if numerical_features.size > 0 else 0}")
        
        return combined_features, feature_names

# Convenience function for easy usage
def extract_features_from_dataframe(df, glove_path=None, embedding_dim=100, use_embeddings=True):
    
    extractor = ImprovedFeatureExtractor(
        embedding_dim=embedding_dim,
        use_embeddings=use_embeddings
    )
    
    features, feature_names = extractor.fit_transform(data=df, glove_path=glove_path)
    return features, feature_names, extractor

# Backward compatibility alias
FeatureExtractor = ImprovedFeatureExtractor
