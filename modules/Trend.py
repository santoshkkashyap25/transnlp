import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.utils.logger import app_logger
from src.utils.helpers import load_data_from_csv
from src.config import PROCESSED_DATA_DIR
import os
import numpy as np

st.title("Cultural Trend Dashboard")
st.write("Visualize shifts in content appeal and thematic relevance over time, offering strategic foresight.")

processed_data_path = os.path.join(PROCESSED_DATA_DIR, 'processed_content_with_clusters.csv')
content_df = load_data_from_csv(processed_data_path)

TOPICS = ['Culture', 'UK', 'Crimes', 'Situational', 'Immigrants', 'Relationships', 'Politics']

if content_df.empty or 'preprocessed_content' not in content_df.columns or 'Year' not in content_df.columns:
    st.error("Processed data not found, or missing 'preprocessed_content' or 'Year' column.")
    app_logger.error("Required columns missing for Cultural Trend Dashboard.")
    st.stop()

content_df['Year'] = pd.to_numeric(content_df['Year'], errors='coerce').fillna(0).astype(int)
content_df = content_df[content_df['Year'] > 0]

for col in TOPICS:
    content_df[col] = pd.to_numeric(content_df[col], errors='coerce').fillna(0)

topic_sum = content_df[TOPICS].sum(axis=1).replace(0, np.nan)
content_df[TOPICS] = content_df[TOPICS].div(topic_sum, axis=0).fillna(0) * 100

# Add dominant topic and its proportion
content_df['dominant_topic'] = content_df[TOPICS].idxmax(axis=1)
content_df['dominant_topic_proportion'] = content_df[TOPICS].max(axis=1)

# IMDb rating column fix
if 'rating' in content_df.columns:
    content_df['imdb_rating'] = pd.to_numeric(content_df['rating'], errors='coerce').fillna(0)
else:
    content_df['imdb_rating'] = 0

st.subheader("Topic Popularity Over Time")
years = sorted(content_df['Year'].unique())
topic_trends = []

# Build topic trend data
for year in years:
    year_data = content_df[content_df['Year'] == year]
    avg_props = year_data[TOPICS].mean().tolist()
    topic_trends.append([year] + avg_props)

if topic_trends:
    topic_trend_df = pd.DataFrame(topic_trends, columns=['Year'] + TOPICS).set_index('Year')

    fig_topic_popularity = go.Figure()
    for topic in TOPICS:
        fig_topic_popularity.add_trace(go.Scatter(
            x=topic_trend_df.index, y=topic_trend_df[topic],
            mode='lines+markers', name=topic
        ))

    fig_topic_popularity.update_layout(
        title='Average Topic Proportion Over Time',
        xaxis_title='Year',
        yaxis_title='Average Proportion (%)',
        hovermode="x unified"
    )
    st.plotly_chart(fig_topic_popularity, use_container_width=True)
    app_logger.info("Topic popularity trend plotted.")
else:
    st.info("Insufficient data to display topic popularity trends.")

st.subheader("Evolving Audience Reception")
st.write("Track how average IMDb ratings for dominant topics change over time.")

if 'imdb_rating' in content_df.columns and 'dominant_topic' in content_df.columns:
    avg_rating_trends = content_df.groupby(['Year', 'dominant_topic'])['imdb_rating'].mean().reset_index()

    selected_trend_topic = st.selectbox(
        "Select a topic to view its IMDb rating trend:",
        options=TOPICS,
        key="trend_topic_select"
    )

    filtered_trend_df = avg_rating_trends[avg_rating_trends['dominant_topic'] == selected_trend_topic]

    if not filtered_trend_df.empty:
        fig_rating_trend = px.line(
            filtered_trend_df,
            x='Year',
            y='imdb_rating',
            title=f'Average IMDb Rating Trend for "{selected_trend_topic}"',
            labels={'Year': 'Release Year', 'imdb_rating': 'Avg IMDb Rating'}
        )
        st.plotly_chart(fig_rating_trend, use_container_width=True)
        app_logger.info(f"IMDb rating trend for topic '{selected_trend_topic}' displayed.")
    else:
        st.info(f"No data available for topic '{selected_trend_topic}'.")
else:
    st.warning("IMDb rating or dominant_topic column missing.")
    app_logger.warning("Cannot show rating trends; column missing.")