import streamlit as st
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from utils.logger import setup_logging
from src.config import APP_NAME

from modules import predict

logger = setup_logging()

st.set_page_config(
    page_title=APP_NAME,
    layout="wide"
)

PAGES = {
    "Success Predictor": predict.app,
    # # "Audience Insights": Insights.app,
    # "Cultural Trend Dashboard": Trend.app,
}

def main_app():
    # choice = st.sidebar.radio("Go to", list(PAGES.keys()))

    # if choice == "Rating Predictor":
    choice = "Success Predictor"
    st.title(f"{APP_NAME}")
    PAGES[choice]()
    logger.info("Main application page loaded.")

if __name__ == "__main__":
    main_app()
