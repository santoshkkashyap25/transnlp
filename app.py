import streamlit as st
import os
import sys
import nltk

# Automatically download required tokenizer if not present
# Safe wrapper to check and download NLTK resources if not already available
def ensure_nltk_resources():
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
    }

    for resource, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource)

# Call this at the beginning of your Streamlit app or any script using NLTK
ensure_nltk_resources()

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from utils.logger import setup_logging
from src.config import APP_NAME

# Setup logging for the application
logger = setup_logging()

st.set_page_config(
    page_title=APP_NAME,
    layout="wide"
)

def main_app():
    st.title(f"{APP_NAME}")
    st.write(
        """
        Welcome to the **Transcript Analysis & IMDb Rating Predictor**!
        This multi-page application transforms raw text into actionable insights for strategic decision-making in media and content creation.
        """
    )

    st.info("Navigate through the pages using the sidebar to explore different functionalities.")

    st.markdown("---")
    st.subheader("How to Use:")
    st.write(
        """
        1. **Content Success Predictor**: Upload content text and get a predicted IMDb rating Type.
        2. **Audience Insights**: Explore audience preferences based on content themes.
        3. **Cultural Trend Dashboard**: Visualize shifts in content appeal and thematic relevance over time.
        """
    )
    st.markdown("---")

    logger.info("Main application page loaded.")

if __name__ == "__main__":
    main_app()