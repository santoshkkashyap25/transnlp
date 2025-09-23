# streamlit_app/src/analytics/sentiment_analysis.py
import pandas as pd
import numpy as np
import math
from textblob import TextBlob
import warnings
import os

from src.utils.helpers import load_data_from_csv
from src.utils.logger import app_logger
from src.config import PROCESSED_DATA_DIR

warnings.filterwarnings('ignore')

def split_text(text, n=10):
    if not isinstance(text, str) or not text:
        app_logger.warning("Attempted to split non-string or empty text.")
        return [""] * n

    length = len(text)
    # Ensure size is at least 1 to avoid ZeroDivisionError if text is very short
    size = max(1, math.floor(length / n))
    
    split_list = []
    for i in range(n):
        start_idx = i * size
        end_idx = min(start_idx + size, length)
        # Ensure don't go out of bounds for the last piece
        if i == n - 1:
            split_list.append(text[start_idx:length])
        else:
            split_list.append(text[start_idx:end_idx])
            
    return split_list


def calculate_sentiment_metrics(data_df):
    app_logger.info("Calculating polarity and subjectivity for transcripts.")
    
    if 'Transcript' not in data_df.columns:
        app_logger.error("'Transcript' column not found in the DataFrame. Cannot calculate sentiment.")
        return data_df

    pol = lambda x: TextBlob(str(x)).sentiment.polarity
    sub = lambda x: TextBlob(str(x)).sentiment.subjectivity
    
    data_df['polarity'] = data_df['Transcript'].apply(pol)
    data_df['subjectivity'] = data_df['Transcript'].apply(sub)
    app_logger.info("Polarity and subjectivity calculation complete.")
    
    return data_df


def get_sentiment_over_time(transcript_text, n_parts=10):

    app_logger.info(f"Analyzing sentiment over time for a transcript, split into {n_parts} parts.")
    
    list_pieces = split_text(transcript_text, n=n_parts)

    polarity_piece_list = []
    for p in list_pieces:
        polarity_piece_list.append(TextBlob(str(p)).sentiment.polarity) # Ensure piece is string
    
    app_logger.info("Sentiment over time calculated for transcript.")
    return polarity_piece_list


if __name__ == "__main__":
    app_logger.info("Running sentiment analysis script directly (functional components only).")

    input_csv_path = os.path.join(PROCESSED_DATA_DIR, 'processed_content_data.csv')
    
    data = load_data_from_csv(input_csv_path)

    if data.empty:
        app_logger.error(f"No data found or loaded from {input_csv_path}. Exiting sentiment analysis script.")
    else:
        data_with_sentiment = calculate_sentiment_metrics(data.copy())
        
        app_logger.info("Sample of DataFrame with overall sentiment metrics:")
        app_logger.info(data_with_sentiment[['Title', 'polarity', 'subjectivity']].head())

        if not data_with_sentiment.empty and 'Transcript' in data_with_sentiment.columns:
            first_transcript = data_with_sentiment.iloc[0]['Transcript']
            first_transcript_title = data_with_sentiment.iloc[0]['Title']
            sentiment_over_time = get_sentiment_over_time(first_transcript)
            app_logger.info(f"Sentiment over time for '{first_transcript_title}': {sentiment_over_time}")
        else:
            app_logger.warning("Cannot demonstrate sentiment over time calculation due to missing data.")

    app_logger.info("Sentiment analysis script finished.")

