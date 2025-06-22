# streamlit_app/src/models/clustering.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # Added for scaling topic vectors
from gensim import corpora, models # Added for Phrases and Dictionary
from src.utils.helpers import load_model, load_data_from_csv
from src.utils.logger import app_logger
from src.config import (
    TFIDF_VECTORIZER_PATH, KMEANS_LDA_PATH, KMEANS_TFIDF_PATH, 
    LDA_MODEL_PATH, TOPICS, PROCESSED_DATA_DIR
)
import numpy as np
import pickle
import os
import spacy # Added for spaCy integration
from src.utils.helpers import identity_tokenizer, identity_analyzer

try:
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    app_logger.info("spaCy model 'en_core_web_sm' loaded successfully in clustering.py.")
except Exception as e:
    app_logger.error(f"Error loading spaCy model in clustering.py: {e}. Please ensure it's downloaded (`python -m spacy download en_core_web_sm`).")
    nlp = None # Set to None if loading fails

def get_cluster_assignments(preprocessed_text):
    """Assigns content to KMeans clusters based on LDA and TF-IDF."""
    tfidf_vectorizer = load_model(TFIDF_VECTORIZER_PATH)
    kmeans_lda = load_model(KMEANS_LDA_PATH)
    kmeans_tfidf = load_model(KMEANS_TFIDF_PATH)
    
    if tfidf_vectorizer is None or kmeans_lda is None or kmeans_tfidf is None:
        app_logger.error("One or more clustering models/vectorizers not loaded. Cannot assign clusters. Please train the models first.")
        return None, None

    # --- TF-IDF Cluster Assignment ---
    words_for_tfidf = preprocessed_text.split() if isinstance(preprocessed_text, str) else []

    processed_for_tfidf = []
    if nlp and words_for_tfidf:
        doc = nlp(" ".join(words_for_tfidf))
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']
        processed_for_tfidf = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
    else:
        # Fallback if spaCy is not loaded, just use original words for TF-IDF
        processed_for_tfidf = words_for_tfidf
        app_logger.warning("Using basic tokenization for TF-IDF as spaCy is not loaded.")

    if not processed_for_tfidf:
        app_logger.warning("Processed text for TF-IDF is empty. Cannot generate TF-IDF cluster.")
        tfidf_cluster = None
    else:
        # The TF-IDF vectorizer was trained with analyzer=lambda x: x and tokenizer=lambda x: x,
        # expecting a list of lists of tokens.
        tfidf_vector = tfidf_vectorizer.transform([processed_for_tfidf]) # Wrap in a list as transform expects iterable of docs
        tfidf_cluster = kmeans_tfidf.predict(tfidf_vector)[0]


    # --- LDA Cluster Assignment ---
    # This requires loading the LDA model and dictionary here, as in the previous version.
    lda_model = load_model(LDA_MODEL_PATH)
    dictionary_path = LDA_MODEL_PATH.replace('.pkl', '_dict.pkl')
    dictionary = load_model(dictionary_path)

    if lda_model is None or dictionary is None:
        app_logger.error("LDA model or dictionary not loaded for cluster assignment. Cannot generate LDA cluster.")
        lda_cluster = None
    else:
        # For LDA, we need the document in bag-of-words format using the original dictionary
        words_for_lda = preprocessed_text.split() if isinstance(preprocessed_text, str) else []
        if not words_for_lda:
            app_logger.warning("Preprocessed text is empty or not a string. Cannot generate LDA topic vectors for clustering.")
            lda_cluster = None # Or handle as per application logic
        else:
            doc_corpus = dictionary.doc2bow(words_for_lda)
            topic_distribution = lda_model.get_document_topics(doc_corpus, minimum_probability=0.0)
            
            lda_topic_vector = np.zeros(len(TOPICS))
            for topic_id, proportion in topic_distribution:
                if topic_id < len(TOPICS):
                    lda_topic_vector[topic_id] = proportion
            
            lda_cluster = kmeans_lda.predict(lda_topic_vector.reshape(1, -1))[0] # Reshape for single sample

    app_logger.info(f"Assigned content to LDA cluster {lda_cluster} and TF-IDF cluster {tfidf_cluster}.")
    return lda_cluster, tfidf_cluster


def train_clustering_models(data_df):
    app_logger.info("Starting clustering model training...")

    tfidf_vectorizer_exists = os.path.exists(TFIDF_VECTORIZER_PATH)
    kmeans_lda_exists = os.path.exists(KMEANS_LDA_PATH)
    kmeans_tfidf_exists = os.path.exists(KMEANS_TFIDF_PATH)

    if tfidf_vectorizer_exists and kmeans_lda_exists and kmeans_tfidf_exists:
        app_logger.info("TF-IDF vectorizer and KMeans models already exist. Skipping training.")
        return

    # Ensure 'preprocessed_content' column exists and is not empty
    if 'preprocessed_content' not in data_df.columns or data_df['preprocessed_content'].empty:
        app_logger.error("'preprocessed_content' column not found or is empty in DataFrame. Cannot train clustering models.")
        return

    texts_for_training = data_df['preprocessed_content'].tolist()

    # Filter out empty strings from texts_for_training before processing
    texts_for_training = [text for text in texts_for_training if isinstance(text, str) and text.strip()]
    if not texts_for_training:
        app_logger.warning("No valid text content for clustering model training after filtering empty strings.")
        return

    # --- Prepare n-grams and load LDA for clustering training ---
    app_logger.info("Building n-gram models for clustering training...")
    tokenized_texts = [doc.split() for doc in texts_for_training]
    
    bigram_phrases = models.Phrases(tokenized_texts, min_count=10)
    trigram_phrases = models.Phrases(bigram_phrases[tokenized_texts], min_count=5)
    bigram_model = models.phrases.Phraser(bigram_phrases)
    trigram_model = models.phrases.Phraser(trigram_phrases)
    trigrams = [trigram_model[bigram_model[doc]] for doc in tokenized_texts]
    app_logger.info("N-grams re-generated for clustering training.")

    app_logger.info("Loading LDA model and dictionary for LDA-based clustering...")
    lda_model = load_model(LDA_MODEL_PATH)
    dictionary_path = LDA_MODEL_PATH.replace('.pkl', '_dict.pkl')
    id2word = load_model(dictionary_path)

    try:
        if lda_model is None or id2word is None:
            app_logger.error("LDA model or dictionary not found. Please train topic models first. Cannot train LDA KMeans.")
            kmeans_lda = None
        else:
            corpus = [id2word.doc2bow(text) for text in trigrams]

            # --- KMeans for LDA topics ---
            app_logger.info("Training KMeans for LDA topics...")
            lda_vectors = []
            num_topics = len(TOPICS)
            for doc_corpus in corpus:
                topic_distribution = lda_model.get_document_topics(doc_corpus, minimum_probability=0.0)
                topic_vector = np.zeros(num_topics)
                for topic_id, proportion in topic_distribution:
                    if topic_id < num_topics:
                        topic_vector[topic_id] = proportion
                lda_vectors.append(topic_vector)
            
            if lda_vectors:
                lda_vectors = np.array(lda_vectors)
                scaler = StandardScaler()
                scaled_lda_vectors = scaler.fit_transform(lda_vectors)

                if not kmeans_lda_exists:
                    kmeans_lda = KMeans(n_clusters=7, random_state=10, n_init=10) 
                    kmeans_lda.fit(scaled_lda_vectors)
                    with open(KMEANS_LDA_PATH, 'wb') as f:
                        pickle.dump(kmeans_lda, f)
                    app_logger.info("KMeans for LDA saved.")
                else:
                    app_logger.info("KMeans for LDA already exists. Skipping training.")
            else:
                app_logger.warning("No LDA vectors generated for KMeans LDA training. Skipping.")
                kmeans_lda = None
    except Exception as e:
        app_logger.error(f"Error during LDA-based clustering setup: {e}")
        kmeans_lda = None


    # --- TF-IDF Vectorizer Training ---
    app_logger.info("Training TF-IDF vectorizer...")
    if not tfidf_vectorizer_exists:
        # Using global identity_tokenizer and identity_analyzer to avoid pickling error
        tfidf_vectorizer = TfidfVectorizer(max_features=1000, tokenizer=identity_tokenizer, 
                                           lowercase=False, min_df=10, max_df=0.4, 
                                           analyzer=identity_analyzer) 
        
        # Prepare lemmatized words for TF-IDF training
        lemmatized_words_for_tfidf_training = []
        if nlp:
            for sent_tokens in trigrams: # trigrams already processed from tokenized_texts
                doc = nlp(" ".join(sent_tokens))
                lemmatized_words_for_tfidf_training.append([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']])
        else:
            # Fallback for TF-IDF training if spaCy not loaded
            lemmatized_words_for_tfidf_training = [list(sent) for sent in trigrams] # Use trigrams as is
            app_logger.warning("SpaCy not loaded for TF-IDF training; using raw trigrams for tokenization.")

        tfidf_vectorizer.fit(lemmatized_words_for_tfidf_training) # Fit on lemmatized trigram texts
        with open(TFIDF_VECTORIZER_PATH, 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        app_logger.info("TF-IDF vectorizer saved.")
    else:
        app_logger.info("TF-IDF vectorizer already exists. Skipping training.")
        tfidf_vectorizer = load_model(TFIDF_VECTORIZER_PATH) # Load if exists to use for KMeans TF-IDF

    # --- KMeans for TF-IDF vectors ---
    app_logger.info("Training KMeans for TF-IDF...")
    if tfidf_vectorizer is not None:
        # Use the same lemmatized words for TF-IDF clustering training
        if 'lemmatized_words_for_tfidf_training' in locals():
            tfidf_vectors = tfidf_vectorizer.transform(lemmatized_words_for_tfidf_training).toarray()
        else:
            lemmatized_words_for_tfidf_training_re = []
            if nlp:
                for sent_tokens in trigrams:
                    doc = nlp(" ".join(sent_tokens))
                    lemmatized_words_for_tfidf_training_re.append([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']])
            else:
                lemmatized_words_for_tfidf_training_re = [list(sent) for sent in trigrams]
            tfidf_vectors = tfidf_vectorizer.transform(lemmatized_words_for_tfidf_training_re).toarray()


        if tfidf_vectors.shape[0] > 0:
            if not kmeans_tfidf_exists:
                # Use n_clusters=7 and random_state=10 as in the notebook's final clustering
                kmeans_tfidf = KMeans(n_clusters=7, random_state=10, n_init=10) 
                kmeans_tfidf.fit(tfidf_vectors)
                with open(KMEANS_TFIDF_PATH, 'wb') as f:
                    pickle.dump(kmeans_tfidf, f)
                app_logger.info("KMeans for TF-IDF saved.")
            else:
                app_logger.info("KMeans for TF-IDF already exists. Skipping training.")
        else:
            app_logger.warning("No TF-IDF vectors generated for KMeans TF-IDF training. Skipping.")
    else:
        app_logger.error("TF-IDF vectorizer not available. Cannot train KMeans for TF-IDF.")

    app_logger.info("Clustering model training and saving complete.")


def add_cluster_columns_to_data():
    """
    Loads data with LDA topic probabilities, applies trained KMeans models for LDA and TF-IDF,
    adds cluster assignments as new columns, and saves the updated DataFrame.
    """
    app_logger.info("Starting process to add cluster columns to data.")

    # Define input and output file paths
    input_csv_path = os.path.join(PROCESSED_DATA_DIR, 'processed_content_with_topics.csv')
    output_csv_path = os.path.join(PROCESSED_DATA_DIR, 'processed_content_with_clusters.csv')

    # Load the data with topic probabilities
    df = load_data_from_csv(input_csv_path)
    if df.empty:
        app_logger.error(f"No data found or loaded from {input_csv_path}. Cannot add cluster columns.")
        return

    # Ensure required columns exist
    topic_columns = TOPICS
    if not all(col in df.columns for col in topic_columns):
        app_logger.error(f"Missing one or more topic columns in DataFrame. Expected: {topic_columns}. Please ensure topic probabilities are added first.")
        return
    if 'preprocessed_content' not in df.columns:
        app_logger.error("'preprocessed_content' column not found in the DataFrame. Cannot perform TF-IDF clustering.")
        return

    kmeans_lda = load_model(KMEANS_LDA_PATH)
    kmeans_tfidf = load_model(KMEANS_TFIDF_PATH)
    tfidf_vectorizer = load_model(TFIDF_VECTORIZER_PATH)
    
    if kmeans_lda is None or kmeans_tfidf is None or tfidf_vectorizer is None:
        app_logger.error("One or more clustering models/vectorizers not loaded. Cannot add cluster columns. Please train the models first.")
        return

    # --- Add cluster_LDA column ---
    app_logger.info("Adding 'cluster_LDA' column...")
    X_lda_topics = df[topic_columns].copy()
    
    # Scale the topic probabilities, as KMeans LDA was trained on scaled data
    scaler_lda = StandardScaler()
    scaled_X_lda_topics = scaler_lda.fit_transform(X_lda_topics)
    
    df['cluster_LDA'] = kmeans_lda.predict(scaled_X_lda_topics)
    app_logger.info("'cluster_LDA' column added.")

    # --- Add cluster_tfidf column ---
    app_logger.info("Adding 'cluster_tfidf' column...")
    
    # Re-apply n-grams and lemmatization to the 'preprocessed_content' for TF-IDF consistency
    texts_for_tfidf_clustering = df['preprocessed_content'].tolist()
    tokenized_texts_for_tfidf = [doc.split() for doc in texts_for_tfidf_clustering]
    
    # Re-generate Phraser models for consistency. In a production app, these might be saved.
    bigram_phrases_for_tfidf = models.Phrases(tokenized_texts_for_tfidf, min_count=10)
    trigram_phrases_for_tfidf = models.Phrases(bigram_phrases_for_tfidf[tokenized_texts_for_tfidf], min_count=5)
    bigram_model_for_tfidf = models.phrases.Phraser(bigram_phrases_for_tfidf)
    trigram_model_for_tfidf = models.phrases.Phraser(trigram_phrases_for_tfidf)
    
    trigrams_for_tfidf = [trigram_model_for_tfidf[bigram_model_for_tfidf[doc]] for doc in tokenized_texts_for_tfidf]

    lemmatized_words_for_tfidf = []
    if nlp:
        for sent in trigrams_for_tfidf:
            doc = nlp(" ".join(sent))
            lemmatized_words_for_tfidf.append([token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']])
    else:
        # Fallback if spaCy is not loaded, just use trigrams
        lemmatized_words_for_tfidf = [list(sent) for sent in trigrams_for_tfidf]
        app_logger.warning("SpaCy not loaded for TF-IDF clustering; using raw trigrams for tokenization.")

    if any(lem_list for lem_list in lemmatized_words_for_tfidf):
        X_tfidf = tfidf_vectorizer.transform(lemmatized_words_for_tfidf)
        df['cluster_tfidf'] = kmeans_tfidf.predict(X_tfidf)
    else:
        df['cluster_tfidf'] = np.nan # Assign NaN or a default if no valid TF-IDF vectors
        app_logger.warning("No valid lemmatized words for TF-IDF clustering. 'cluster_tfidf' set to NaN.")

    app_logger.info("'cluster_tfidf' column added.")

    try:
        df.to_csv(output_csv_path, index=False)
        app_logger.info(f"DataFrame with cluster columns saved successfully to {output_csv_path}.")
    except Exception as e:
        app_logger.error(f"Error saving DataFrame to CSV at {output_csv_path}: {e}")

if __name__ == "__main__":
    app_logger.info("Running clustering model script directly.")
    processed_data_file_for_training = os.path.join(PROCESSED_DATA_DIR, 'processed_content_data.csv')
    df_train = load_data_from_csv(processed_data_file_for_training)
    if not df_train.empty:
        train_clustering_models(df_train)
        app_logger.info("Clustering models trained and saved successfully (if not already present).")
    else:
        app_logger.error(f"No data found at {processed_data_file_for_training} for training. Please ensure data is scraped and preprocessed.")


    add_cluster_columns_to_data()
    app_logger.info("Script for adding cluster columns finished.")
