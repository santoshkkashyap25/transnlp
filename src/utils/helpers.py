import os
import pickle
import joblib
import streamlit as st
import logging
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

app_logger = logging.getLogger(__name__)

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    app_logger.info("Downloading NLTK data...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    app_logger.info("NLTK data downloaded.")

def clean_text(text):
    """Removes punctuation and non-alphabetic characters."""
    text = text.lower()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove non-alphabetical characters
    return text

def lemmatize_text(text):
    """Lemmatizes text."""
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def filter_pos(text, allowed_pos=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """Filters tokens by Parts-of-Speech."""
    allowed_tags = [p[0] for p in allowed_pos] # e.g., 'N', 'J', 'V', 'R'
    tokens = word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    filtered_tokens = [
        word for word, tag in tagged_tokens
        if tag[:1] in allowed_tags
    ]
    return ' '.join(filtered_tokens)


def remove_stopwords(text):
    """Removes common English stopwords."""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

# These functions are used by the pickled models.
def identity_tokenizer(text):
    return text

def identity_analyzer(text):
    return text


@st.cache_resource
def load_model(model_path):
    """Loads a pre-trained model from a given path (.pkl or .joblib supported)."""
    if not os.path.exists(model_path):
        app_logger.error(f"Model file not found: {model_path}")
        st.error(
            f"Error: Required model file not found at {model_path}. "
            "Please ensure models are placed in the 'data/models' directory."
        )
        return None

    try:
        if model_path.endswith(".pkl"):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        elif model_path.endswith(".joblib"):
            model = joblib.load(model_path)
        else:
            raise ValueError("Unsupported file format. Must be .pkl or .joblib")

        app_logger.info(f"Model loaded successfully from {model_path}")
        return model

    except Exception as e:
        app_logger.error(f"Error loading model from {model_path}: {e}")
        st.error(f"Error loading model from {model_path}. Details: {e}")
        return None


@st.cache_data
def load_data_from_csv(file_path):
    """Loads data from a given CSV path."""
    if not os.path.exists(file_path):
        app_logger.warning(f"Data file not found: {file_path}.")
        st.warning(f"Data file not found at {file_path}.")
        return pd.DataFrame() # Return empty DataFrame
    else:
        app_logger.info(f"Loading data from {file_path}")
        return pd.read_csv(file_path)