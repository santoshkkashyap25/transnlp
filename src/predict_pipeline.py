import numpy as np
import pandas as pd
from src.models.topic_modelling import apply_nlp_pipeline, get_topic_vectors
from src.utils.helpers import load_model
from src.config import ENSEMBLE_CLASSIFIER_PATH, MODELS_DIR, TOPICS
import joblib
import os
from src.models.clustering import get_cluster_assignments
from src.utils.helpers import identity_tokenizer, identity_analyzer
from src.utils.logger import app_logger
# Load trained models
ensemble_model = load_model(ENSEMBLE_CLASSIFIER_PATH)
scaler_path = os.path.join(MODELS_DIR, 'production_scaler.joblib')
scaler = joblib.load(scaler_path)

assert ensemble_model is not None, "Ensemble model not loaded."
assert scaler is not None, "Scaler not loaded."

import re
import string
from nltk.tokenize import word_tokenize

def load_swear_words(filepath: str = r"D:\PROJECTS\transnlp\data\profanity.txt") -> set[str]:
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            words = {line.strip().lower() for line in file if line.strip()}
        return words
    except Exception as e:
        app_logger.warning(f"Could not load profanity list: {e}")
        return set()
SWEAR_WORDS = load_swear_words()

def clean_user_transcript(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    # --- Basic text normalization ---
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[‘’“”…♪)(]', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # --- Tokenize and remove profanity ---
    tokens = word_tokenize(text)
    clean_tokens = [token for token in tokens if token not in SWEAR_WORDS]

    return ' '.join(clean_tokens)

# Feature columns (same order as training)
X_COLUMNS = [
    'Culture', 'UK', 'Crimes', 'Situational', 'Immigrants', 'Relationships', 'Politics',
    '0_LDA', '1_LDA', '2_LDA', '3_LDA', '4_LDA', '5_LDA', '6_LDA',
    '0_tfidf', '1_tfidf', '2_tfidf', '3_tfidf', '4_tfidf', '5_tfidf', '6_tfidf'
]

def predict_rating_type(transcript: str):
    cleaned_input = clean_user_transcript(transcript)
    # Step 1: Preprocess the transcript
    preprocessed_text = apply_nlp_pipeline(cleaned_input)

    # Step 2: Get topic distribution from LDA model
    topic_vector = get_topic_vectors(preprocessed_text)
    if topic_vector is None:
        raise ValueError("Could not extract topic vector.")

    # Step 3: Get actual cluster one-hot vectors
    lda_cluster, tfidf_cluster = get_cluster_assignments(preprocessed_text)
    if lda_cluster is None or tfidf_cluster is None:
        raise ValueError("Could not determine LDA or TF-IDF cluster.")
    
    lda_onehot = [1 if i == lda_cluster else 0 for i in range(7)]
    tfidf_onehot = [1 if i == tfidf_cluster else 0 for i in range(7)]

    # Step 4: Combine features and scale
    full_vector = topic_vector.tolist() + lda_onehot + tfidf_onehot
    input_array = np.array(full_vector).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # Step 5: Predict
    pred_encoded = ensemble_model.predict(input_scaled)[0]
    pred_proba = ensemble_model.predict_proba(input_scaled)[0]
    predicted_class = "Above Average" if pred_encoded == 1 else "Below Average"

    return predicted_class, pred_proba



if __name__ == "__main__":
    user_input = "This is a hilarious stand-up piece that touches on politics and relationships in the UK."
    result_class, probabilities = predict_rating_type(user_input)
    print("Prediction:", result_class)
    print("Probabilities:", probabilities)
