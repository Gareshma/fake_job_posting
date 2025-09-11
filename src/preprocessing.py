import pandas as pd
import numpy as np
import re
from collections import Counter

class DataPreprocessor:
    def __init__(self):
        self.drops = []
        self.categories = []
        
    def load_data(self, filepath):
        
        self.data = pd.read_csv(filepath)
        print(f"Dataset shape: {self.data.shape}")
        return self.data
    
    def basic_preprocessing(self):
        print(f"Datatype of each column:\n{self.data.dtypes}")
        print(f"Unique values in each column:\n{self.data.nunique()}")
        
        # Identify categorical attributes
        class_label = "fraudulent"
        nuniq = self.data.nunique().to_dict()
        
        for k, v in nuniq.items():
            if k == class_label:
                continue
            if v < 50:
                self.categories.append(k)
        
        print(f"Categorical Attributes: {self.categories}")
        
        # Identify columns to drop
        self.drops = ["job_id"]
        columns = self.data.columns
        
        for col in columns:
            missing = self.data[col].isna().sum()
            print(f"{col}: {missing} missing values")
            if missing >= len(self.data) / 2:
                self.drops.append(col)
        
        print(f"Columns to drop: {self.drops}")
        self.data = self.data.drop(self.drops, axis=1)
        
        return self.data
    
    def process_location(self):
        
        if 'location' in self.data.columns:
            location_split = self.data['location'].str.split(',', expand=True)
            location_split = location_split.reindex(columns=[0, 1, 2])
            location_split.columns = ['location_country', 'location_state', 'location_city']
            location_split = location_split.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
            
            self.data = self.data.join(location_split)
            self.data = self.data.drop(["location"], axis=1)
            
            self.categories.append("location_country")
            self.categories = list(set(self.categories))
        
        return self.data
    
    def clean_text_columns(self):
        
        text_columns = ['title', 'description', 'requirements', 'benefits', 'company_profile']
        
        for col in text_columns:
            if col in self.data.columns:
                # Fill NaN with empty string
                self.data[col] = self.data[col].fillna('')
                
                # Basic text cleaning
                self.data[col] = self.data[col].apply(self._clean_text)
        
        return self.data
    
    def _clean_text(self, text):
        
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_class_distribution(self):
        
        Y = self.data["fraudulent"]
        ctr = Counter(Y)
        print(f"Class distribution: {ctr}")
        print(f"Accuracy of majority classifier: {ctr[0] / len(self.data):.4f}")
        return ctr
    
    def prepare_features_target(self):
        
        # Combine important text fields
        text_cols = ['title', 'description', 'requirements', 'benefits', 'company_profile']
        existing_text_cols = [col for col in text_cols if col in self.data.columns]
        
        # Create combined text feature
        if existing_text_cols:
            self.data['combined_text'] = self.data[existing_text_cols].apply(
                lambda row: ' '.join(row.values.astype(str)), axis=1
            )
        
        # Separate categorical and numerical features
        categorical_features = [col for col in self.categories if col in self.data.columns]
        numerical_features = [col for col in self.data.columns 
                            if col not in categorical_features + ['fraudulent', 'combined_text'] + existing_text_cols]
        
        X_categorical = self.data[categorical_features] if categorical_features else pd.DataFrame()
        X_numerical = self.data[numerical_features] if numerical_features else pd.DataFrame()
        X_text = self.data['combined_text'] if 'combined_text' in self.data.columns else pd.Series([''] * len(self.data))
        y = self.data['fraudulent']
        
        print(f"Categorical features: {categorical_features}")
        print(f"Numerical features: {numerical_features}")
        print(f"Text feature created: combined_text")
        
        return X_categorical, X_numerical, X_text, y