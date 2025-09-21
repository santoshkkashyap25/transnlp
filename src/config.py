import os

# Application constants
APP_NAME = "Transcript Success Predictor"

# Data and Model Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

# Model file names
LDA_MODEL_PATH = os.path.join(MODELS_DIR, 'lda_model.pkl')
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
KMEANS_LDA_PATH = os.path.join(MODELS_DIR, 'kmeans_lda.pkl')
KMEANS_TFIDF_PATH = os.path.join(MODELS_DIR, 'kmeans_tfidf.pkl')
ENSEMBLE_CLASSIFIER_PATH = os.path.join(MODELS_DIR, 'ensemble_classifier.joblib')


# NLP Constants
TOPICS = [
    'Culture',
    'UK',
    'Crimes',
    'Situational',
    'Immigrants',
    'Relationships',
    'Politics'
]
# Scraping Configuration
SCRAPING_BASE_URL = "https://scrapsfromtheloft.com/stand-up-comedy-scripts/"
TRANSCRIPTS_RAW_DIR = os.path.join(RAW_DATA_DIR, 'transcripts') # New directory for raw transcripts

LOG_FILE_PATH = "app_logs.log"