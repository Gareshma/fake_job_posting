# Fake Job Posting Detection System  

A machine learning system to classify job postings as **real** or **fake**, helping reduce risks such as identity theft and fraud on job platforms.  
The project uses **GloVe embeddings**, **XGBoost**, and **One-Class SVM**, and is deployed as a **Flask web application**.  

---

## Features
- Preprocessing pipeline for job descriptions and metadata (`src/preprocessing.py`)  
- Feature extraction using **GloVe embeddings** (`src/feature_extraction.py`)  
- ML models: **XGBoost** and **One-Class SVM** (`src/models.py`)  
- Evaluation and visualization scripts (`src/evaluation.py`, `/results`)  
- Flask web app (`app.py`) with REST API and UI  
- Sample jobs for quick testing (`sample_jobs.py`)  
- Ready-to-use trained models stored in `/trained_models`  

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/gareshma/fake-job-posting.git
cd fake-job-posting
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

### 3. Download GloVe embeddings  
This project requires **GloVe 100d embeddings**.  
Download from: [GloVe official site](https://nlp.stanford.edu/projects/glove/)  
Place the file `glove.6B.100d.txt` inside the `embeddings/` folder.

### 4. Run the Flask app
```bash
python app.py
```
Open your browser at:  
http://127.0.0.1:5000  

### 5. Run evaluation
```bash
python src/evaluation.py
```

---

## Results  

- **XGBoost Accuracy:** 97.5%  
- **F1 Score:** 0.845  
- Evaluation plots are available in `/results`.  

---

## Future Improvements
- Add transformer-based embeddings (BERT, RoBERTa)  
- Expand dataset with real-world postings  
- Containerize with Docker for scalable deployment  

---

## Author  

**Gareshma Nagalapatti**  
Master’s in Computer Science | Santa Clara University  