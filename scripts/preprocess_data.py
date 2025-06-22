# In streamlit_app/scripts/preprocess_data.py

import pandas as pd
import numpy as np
import os
from src.utils.logger import app_logger
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.models.topic_modelling import apply_nlp_pipeline
from nltk import word_tokenize

def preprocess_data():
    app_logger.info("Starting data preprocessing (after scraping).")
    scraped_data_path = os.path.join(RAW_DATA_DIR, 'scraped_and_cleaned_content_data.csv')

    if not os.path.exists(scraped_data_path):
        app_logger.error(f"Scraped data file not found: {scraped_data_path}. Please run scrape_data.py first.")
        return None

    df = pd.read_csv(scraped_data_path)
    app_logger.info("Applying NLP pipeline to 'Transcript' column.")
    if 'Transcript' in df.columns and not df['Transcript'].empty:
        df['preprocessed_content'] = df['Transcript'].apply(lambda x: apply_nlp_pipeline(x) if pd.notna(x) else '')
    else:
        app_logger.warning("No 'Transcript' column or it is empty. 'preprocessed_content' will be empty.")
        df['preprocessed_content'] = ''

    # Calculate Rating Type (dynamic threshold based on dataset mean)
    app_logger.info("Calculating rating type based on mean rating.")
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    mean_rating = df['rating'].mean() # Calculate mean from the current DataFrame
    # Assign 'Above Average' if rating is at or above mean, 'Below Average' otherwise
    df['rating_type'] = np.where(df['rating'] >= mean_rating, 'Above Average', 'Below Average')
    df['rating_type'] = np.where(pd.isna(df['rating']), np.nan, df['rating_type']) # Handle NaN ratings

    # Tokenize preprocessed_content for further calculations (word counts, diversity, profanity)
    df['temp_tokens'] = df['preprocessed_content'].apply(lambda x: word_tokenize(x) if pd.notna(x) else [])

    # Profanity detection and removal
    app_logger.info("Detecting and removing profanity words.")
    f_words = ['fuck', 'fucking', 'fckin','fucken','fucked','fck','fcking','fuckin', 'fucker', 'muthafucka', 'motherfuckers', 'motherfucke','motha','motherfucking','motherfuckin','motherfuckers', 'motherfucker']
    s_words = ['shit', 'shitter', 'shitting', 'shite', 'bullshit', 'shitty']
    # Combined list of swear words
    swears = f_words + s_words + ['cunt', 'asshole', 'damn', 'goddamn', 'cocksucker','sluts','dicks','dick','pussy','ass','asshole','assholes','porn','penis','tit']

    # Helper function to count specific swear words
    def get_swear_counts(token_list, swear_list):
        swears_count = 0
        for word in token_list:
            if word.lower() in swear_list:
                swears_count += 1
        return swears_count

    df['f_words'] = df['temp_tokens'].apply(lambda x: get_swear_counts(x, f_words))
    df['s_words'] = df['temp_tokens'].apply(lambda x: get_swear_counts(x, s_words))

    # Filter out all identified swear words from the tokens
    df['final_tokens'] = df['temp_tokens'].apply(lambda x: [word for word in x if word.lower() not in swears])

    # Recalculate word_count and diversity using the final_tokens (after profanity removal)
    app_logger.info("Calculating word count, diversity, and diversity ratio after profanity removal.")
    df['word_count'] = df['final_tokens'].apply(lambda x: len(x))
    df['diversity'] = df['final_tokens'].apply(lambda x: len(set(x)))
    df['diversity_ratio'] = df.apply(lambda row: row['diversity'] / row['word_count'] if row['word_count'] > 0 else 0, axis=1)

    df['preprocessed_content'] = df['final_tokens'].apply(lambda x: ' '.join(x))

    # Drop temporary token columns
    df = df.drop(columns=['temp_tokens', 'final_tokens'])

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, 'processed_content_data.csv')
    df.to_csv(output_path, index=False)
    app_logger.info(f"Processed data saved to: {output_path}")
    return df

if __name__ == "__main__":
    preprocess_data()