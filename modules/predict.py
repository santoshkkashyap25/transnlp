import streamlit as st
import pandas as pd
import numpy as np
from src.nltk_setup import setup_nltk_data
from src.utils.logger import app_logger

try:
    from src.predict_pipeline import predict_rating_type
except ImportError:
    import time
    time.sleep(2)
    from src.predict_pipeline import predict_rating_type

from src.utils.helpers import identity_tokenizer, identity_analyzer


def app():
    st.set_page_config(page_title="Transcript Success Predictor", layout="wide")
    st.write("Predict the potential success of new content based on its textual content.")

    # --- Input Section ---
    st.subheader("Input Transcript")
    input_text = st.text_area(
        "Paste your  transcript, early draft, or synopsis here:",
        height=300,
        placeholder="e.g., 'This comedy special explores the absurdity of modern politics and ...'",
        key="content_input_text"
    )

    uploaded_file = st.file_uploader("Or, upload a text file (.txt)", type=["txt"], key="file_upload")

    content_to_analyze = ""
    if uploaded_file is not None:
        try:
            content_to_analyze = uploaded_file.read().decode("utf-8")
            st.write("File uploaded successfully!")
            app_logger.info(f"User uploaded file: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            app_logger.error(f"Could not read uploaded file {uploaded_file.name}: {e}")
    elif input_text:
        content_to_analyze = input_text

    if st.button("Predict Success", key="predict_button"):
        if content_to_analyze and len(content_to_analyze.strip()) >= 50:
            with st.spinner("Analyzing transcript and predicting success..."):
                try:
                    # trigger the model loading
                    predicted_category, prediction_proba = predict_rating_type(content_to_analyze)

                    st.markdown("---")
                    st.subheader("Prediction Results")

                    if predicted_category == "Above Average":
                        st.success(f"**Predicted Success: {predicted_category} 🎉**")
                    else:
                        st.warning(f"**Predicted Success: {predicted_category} ⚠️**")

                    # probability bar chart
                    if prediction_proba is not None:
                        st.write("Prediction Confidence:")
                        prob_df = pd.DataFrame({
                            'Category': ['Below Average', 'Above Average'],
                            'Probability': prediction_proba
                        })
                        st.bar_chart(prob_df.set_index('Category'))

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    app_logger.error(f"Prediction pipeline failed: {e}", exc_info=True)

        elif not content_to_analyze:
            st.warning("Please enter or upload content to predict its success.")
            app_logger.warning("User attempted prediction without providing content.")
        else:  # Text is too short
            st.warning("Please enter a more substantial text for analysis (at least 50 characters).")
            app_logger.warning("User attempted prediction with very short text.")
