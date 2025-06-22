# streamlit_app/src/models/topic_modeling.py
import pandas as pd
from gensim import corpora, models
from src.utils.helpers import clean_text, lemmatize_text, filter_pos, remove_stopwords, load_model, load_data_from_csv
from src.utils.logger import app_logger
from src.config import (
    LDA_MODEL_PATH, TOPICS, PROCESSED_DATA_DIR
)
import numpy as np
import pickle
import os
import spacy

try:
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    app_logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
except Exception as e:
    app_logger.error(f"Error loading spaCy model: {e}. Please ensure it's downloaded (`python -m spacy download en_core_web_sm`).")
    nlp = None # Set to None if loading fails

def apply_nlp_pipeline(text):
    if not isinstance(text, str):
        return "" # Handle non-string inputs gracefully

    cleaned = clean_text(text) # Uses NLTK-based clean_text from helpers

    # Process with spaCy for lemmatization and POS filtering
    if nlp:
        doc = nlp(cleaned)
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
        lemmatized_and_pos_filtered_tokens = [
            token.lemma_ for token in doc if token.pos_ in allowed_postags
        ]
        # Re-join for remove_stopwords, which expects a string
        spacy_processed_text = ' '.join(lemmatized_and_pos_filtered_tokens)
    else:
        # Fallback to NLTK based lemmatize_text and filter_pos if spaCy failed to load
        lemmatized = lemmatize_text(cleaned)
        spacy_processed_text = filter_pos(lemmatized)
        app_logger.warning("Using NLTK fallback for lemmatization/POS filtering as spaCy is not loaded.")

    final_text = remove_stopwords(spacy_processed_text) # Uses NLTK-based remove_stopwords from helpers
    app_logger.info(f"NLP pipeline applied to text.")
    return final_text

def get_topic_vectors(preprocessed_text):
    """Generates LDA topic vectors for the input text."""
    lda_model = load_model(LDA_MODEL_PATH)
    # The dictionary for gensim LDA is typically saved as a separate .dict file or similar
    dictionary_path = LDA_MODEL_PATH.replace('.pkl', '_dict.pkl')
    dictionary = load_model(dictionary_path)

    if lda_model is None or dictionary is None:
        app_logger.error("LDA model or dictionary not loaded. Cannot generate topic vectors. Please train the models first.")
        return None

    # Ensure preprocessed_text is a string, then split it into words
    words = preprocessed_text.split() if isinstance(preprocessed_text, str) else []
    if not words:
        app_logger.warning("Preprocessed text is empty or not a string. Cannot generate topic vectors.")
        return np.zeros(len(TOPICS)) # Return zero vector if no content

    corpus = [dictionary.doc2bow(words)]
    topic_distribution = lda_model.get_document_topics(corpus[0], minimum_probability=0.0) # Ensure all topics are included
    
    topic_vector = np.zeros(len(TOPICS))
    for topic_id, proportion in topic_distribution:
        if topic_id < len(TOPICS): # Ensure topic_id is within expected range
            topic_vector[topic_id] = proportion
    app_logger.info(f"Generated LDA topic vectors.")
    return topic_vector

def train_topic_model(data_df):
    app_logger.info("Starting LDA topic model training...")

    lda_model_exists = os.path.exists(LDA_MODEL_PATH)
    dict_exists = os.path.exists(LDA_MODEL_PATH.replace('.pkl', '_dict.pkl'))

    if lda_model_exists and dict_exists:
        app_logger.info("LDA model and dictionary already exist. Skipping training.")
        return

    # Ensure 'preprocessed_content' column exists and is not empty
    if 'preprocessed_content' not in data_df.columns or data_df['preprocessed_content'].empty:
        app_logger.error("'preprocessed_content' column not found or is empty in DataFrame. Cannot train LDA model.")
        return

    texts_for_training = data_df['preprocessed_content'].tolist()

    # Filter out empty strings from texts_for_training before processing
    texts_for_training = [text for text in texts_for_training if isinstance(text, str) and text.strip()]
    if not texts_for_training:
        app_logger.warning("No valid text content for LDA model training after filtering empty strings.")
        return

    # --- Gensim Bigram and Trigram ---
    app_logger.info("Building bigram and trigram models for LDA training...")
    tokenized_texts = [doc.split() for doc in texts_for_training]

    bigram_phrases = models.Phrases(tokenized_texts, min_count=10)
    trigram_phrases = models.Phrases(bigram_phrases[tokenized_texts], min_count=5)
    bigram_model = models.phrases.Phraser(bigram_phrases)
    trigram_model = models.phrases.Phraser(trigram_phrases)

    # Apply trigram model to texts for LDA
    trigrams = [trigram_model[bigram_model[doc]] for doc in tokenized_texts]
    app_logger.info("Generated n-grams for text data for LDA.")

    # --- LDA Model Training ---
    app_logger.info("Training LDA model...")
    id2word = corpora.Dictionary(trigrams)
    id2word.filter_extremes(no_below=10, no_above=0.4)
    id2word.compactify() # Remove gaps in ids after filtering
    corpus = [id2word.doc2bow(text) for text in trigrams]

    num_topics = len(TOPICS) # Use the number of topics defined in config
    lda_model = models.LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        random_state=1,
        chunksize=30,
        passes=40,
        alpha=0.5,
        eta=0.91,
        eval_every=1,
        per_word_topics=True,
        workers=2 # Adjust based on CPU cores
    )

    # Save LDA model and dictionary
    with open(LDA_MODEL_PATH, 'wb') as f:
        pickle.dump(lda_model, f)
    with open(LDA_MODEL_PATH.replace('.pkl', '_dict.pkl'), 'wb') as f:
        pickle.dump(id2word, f)
    app_logger.info("LDA model and dictionary saved.")
    app_logger.info("LDA topic model training complete.")

if __name__ == "__main__":
    app_logger.info("Running LDA topic model training script directly.")
    
    processed_data_file = os.path.join(PROCESSED_DATA_DIR, 'processed_content_data.csv')
    
    # Load processed data
    df = load_data_from_csv(processed_data_file)
    
    if not df.empty:
        # Train LDA model
        train_topic_model(df)
        app_logger.info("LDA models trained and saved successfully (if not already present).")
    else:
        app_logger.error(f"No data found at {processed_data_file}. Please ensure data is scraped and preprocessed before running NLP model training.")

