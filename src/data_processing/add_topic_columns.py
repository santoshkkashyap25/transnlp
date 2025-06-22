# streamlit_app/src/data_processing/add_topic_columns.py
import pandas as pd
import numpy as np
import os
from gensim import corpora, models 
from src.utils.helpers import load_model, load_data_from_csv
from src.utils.logger import app_logger
from src.config import (
    PROCESSED_DATA_DIR, LDA_MODEL_PATH, TOPICS
)

def add_lda_topic_columns_to_data():
    """
    Loads preprocessed data, applies the trained LDA model to derive topic probabilities,
    adds these as new columns to the DataFrame, and saves the updated DataFrame.
    """
    app_logger.info("Starting process to add LDA topic columns to data.")

    # Define input and output file paths
    input_csv_path = os.path.join(PROCESSED_DATA_DIR, 'processed_content_data.csv')
    output_csv_path = os.path.join(PROCESSED_DATA_DIR, 'processed_content_with_topics.csv')

    # Load the processed data
    df = load_data_from_csv(input_csv_path)
    if df.empty:
        app_logger.error(f"No data found or loaded from {input_csv_path}. Cannot add topic columns.")
        return

    # Ensure 'preprocessed_content' column exists
    if 'preprocessed_content' not in df.columns:
        app_logger.error("'preprocessed_content' column not found in the DataFrame. Please ensure data is preprocessed correctly.")
        return

    # Load the trained LDA model and its dictionary
    lda_model = load_model(LDA_MODEL_PATH)
    dictionary_path = LDA_MODEL_PATH.replace('.pkl', '_dict.pkl')
    id2word = load_model(dictionary_path)

    if lda_model is None or id2word is None:
        app_logger.error("LDA model or dictionary not loaded. Please ensure topic models are trained before running this script.")
        return

    num_topics = len(TOPICS)
    topic_vecs = []

    app_logger.info("Calculating LDA topic probabilities for each document...")
    # Iterate through preprocessed content to get topic distributions
    for index, row in df.iterrows():
        text_content = row['preprocessed_content']
        if pd.isna(text_content) or not isinstance(text_content, str) or not text_content.strip():
            # Handle empty or invalid preprocessed content
            topic_vec = np.zeros(num_topics).tolist()
            app_logger.warning(f"Empty or invalid preprocessed_content for row {index}. Assigning zero vector.")
        else:
            # Convert text content to list of words (assuming space-separated)
            words = text_content.split()
            # Convert the document (list of words) to a bag-of-words (BoW) format
            # This requires the dictionary used during LDA training
            corpus_doc = id2word.doc2bow(words)
            
            # Get topic probabilities for the current document
            # minimum_probability=0.0 ensures all topics are returned, even if probability is very low
            topic_distribution = lda_model.get_document_topics(corpus_doc, minimum_probability=0.0)
            
            # Create a vector of topic probabilities
            topic_vec = np.zeros(num_topics)
            for topic_id, proportion in topic_distribution:
                if topic_id < num_topics: # Ensure topic_id is within the expected range
                    topic_vec[topic_id] = proportion
            topic_vec = topic_vec.tolist() # Convert to list for DataFrame

        topic_vecs.append(topic_vec)

    # Create a DataFrame from the topic vectors
    topic_columns_df = pd.DataFrame(data=topic_vecs, columns=TOPICS, index=df.index)

    # Concatenate the new topic columns with the original DataFrame
    df_with_topics = pd.concat([df, topic_columns_df], axis=1)

    # Save the updated DataFrame to a new CSV file
    try:
        df_with_topics.to_csv(output_csv_path, index=False)
        app_logger.info(f"DataFrame with LDA topic columns saved successfully to {output_csv_path}.")
    except Exception as e:
        app_logger.error(f"Error saving DataFrame to CSV at {output_csv_path}: {e}")

if __name__ == "__main__":
    app_logger.info("Running script to add LDA topic columns to data directly.")
    add_lda_topic_columns_to_data()
    app_logger.info("Script for adding LDA topic columns finished.")
