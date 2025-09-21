import streamlit as st
import pandas as pd
import plotly.express as px
from src.utils.logger import app_logger
from src.utils.helpers import load_data_from_csv
from src.config import PROCESSED_DATA_DIR
import os
import numpy as np

st.title("Audience Insights & Personalization")
st.write("Understand nuanced audience preferences based on content themes for tailored recommendations and marketing.")

# Load processed data
processed_data_path = os.path.join(PROCESSED_DATA_DIR, 'processed_content_with_topics.csv')
content_df = load_data_from_csv(processed_data_path)

# Define actual topic columns from your CSV
topic_columns = ['Culture', 'UK', 'Crimes', 'Situational', 'Immigrants', 'Relationships', 'Politics']

if content_df.empty or 'preprocessed_content' not in content_df.columns:
    st.error("Processed data not found or missing 'preprocessed_content' column. Please run 'Run Data Scraping & Train Models' on the 'Content Success Predictor' page first.")
    app_logger.error("Processed data not available or incomplete for Audience Insights page.")
else:
    # --- Validate topic columns ---
    missing_cols = [col for col in topic_columns if col not in content_df.columns]
    if missing_cols:
        st.error(f"Missing topic columns: {missing_cols}")
        app_logger.error(f"Missing topic columns: {missing_cols}")
        st.stop()

    # --- Convert topic columns to numeric ---
    for col in topic_columns:
        content_df[col] = pd.to_numeric(content_df[col], errors='coerce')

    # --- Check for remaining string values (diagnostics) ---
    non_numeric_counts = content_df[topic_columns].applymap(lambda x: isinstance(x, str)).sum()
    if non_numeric_counts.sum() > 0:
        st.warning(f"Some topic columns still contain non-numeric data: {non_numeric_counts}")
        app_logger.warning(f"Non-numeric data remains in topic columns: {non_numeric_counts}")
        content_df[topic_columns] = content_df[topic_columns].apply(pd.to_numeric, errors='coerce')

    # Fill NaNs with 0
    content_df[topic_columns] = content_df[topic_columns].fillna(0)

    # Normalize to percentages
    topic_sum = content_df[topic_columns].sum(axis=1).replace(0, np.nan)
    content_df[topic_columns] = content_df[topic_columns].div(topic_sum, axis=0).fillna(0) * 100

    # Dominant topic and its proportion
    content_df['dominant_topic'] = content_df[topic_columns].idxmax(axis=1)
    content_df['dominant_topic_proportion'] = content_df[topic_columns].max(axis=1)

    # IMDb Rating conversion
    content_df['imdb_rating'] = pd.to_numeric(content_df.get('rating', 0), errors='coerce').fillna(0)

    st.subheader("Comedian Profiling & Thematic Grouping")
    st.write("This section shows how comedians or content pieces are grouped by dominant topics based on actual data.")

    # Topic Filter
    selected_topic = st.selectbox(
        "Filter content by dominant topic:",
        options=["All"] + topic_columns,
        key="topic_filter"
    )

    filtered_df = content_df.copy()
    if selected_topic != "All":
        filtered_df = content_df[content_df['dominant_topic'] == selected_topic]
        st.write(f"Showing content where '{selected_topic}' is the dominant theme.")
        app_logger.info(f"Filtered content by topic: {selected_topic}")
    else:
        st.write("Showing all content.")
        app_logger.info("Showing all content for audience insights.")

    if not filtered_df.empty:
        display_cols = ['Names', 'Title', 'dominant_topic', 'dominant_topic_proportion', 'imdb_rating']
        display_cols = [col for col in display_cols if col in filtered_df.columns]
        st.dataframe(filtered_df[display_cols], use_container_width=True)

        st.subheader("Audience Segment Mapping")
        st.write("Visualizing how different dominant topics perform based on IMDb rating and topic strength. This helps uncover potential audience clusters.")

        fig = px.scatter(
            filtered_df,
            x='imdb_rating',
            y='dominant_topic_proportion',
            color='dominant_topic',
            size='imdb_rating',
            hover_name='Title',
            title='Content Performance by Dominant Topic',
            labels={'imdb_rating': 'IMDb Rating', 'dominant_topic_proportion': 'Dominant Topic Proportion (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
        app_logger.info("Audience segment mapping chart displayed.")

    else:
        st.info("No content found for the selected filter. Try 'All' or re-check your data.")
