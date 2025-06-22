# TransNLP: Transcript Analysis & IMDb Rating Predictor

[Visit the App](https://transnlp.streamlit.app/)

This project offers a comprehensive platform for analyzing stand-up comedy transcripts, leveraging Natural Language Processing (NLP) to extract valuable insights, predict content success, and visualize cultural trends. Transcripts are sourced from [**Scraps From The Loft**](https://scrapsfromtheloft.com/).

## Features

* **Web Scraping**: Automated collection of stand-up comedy transcripts, comedian names, show titles, and associated metadata (Year, IMDb Rating, Runtime) directly from the web.

* **Data Preprocessing**: Robust cleaning pipeline including text normalization, stopword removal, lemmatization (using spaCy), profanity detection, and feature engineering (word count, diversity).

* **Topic Modeling (LDA)**: Utilizes Latent Dirichlet Allocation to uncover prevalent themes within comedy content (e.g., Politics, Relationships, Culture).

* **Content Clustering**: Groups comedy specials into thematic clusters based on their textual content, providing macro-level insights.

* **IMDb Rating Prediction**: Predicts the potential IMDb rating category (Above Average / Below Average) for new content using an ensemble classification model, powered by extracted NLP features.

* **Trend Analysis**: Visualizes shifts in topic popularity and audience reception over time.

* **Audience Insights**: Helps understand audience preferences by mapping content themes to historical performance.

* **Interactive Streamlit Dashboard**: A user-friendly web application for real-time content analysis, prediction, and interactive data exploration.

## Project Structure
```
transnlp/
├── .streamlit/             # Streamlit specific configuration (e.g., config.toml)
│   └── config.toml
├── data/                   # Centralized data storage
│   ├── raw/                # Raw scraped data and temporary transcript files
│   │   ├── scraped_and_cleaned_content_data.csv
│   │   └── transcripts_raw/ # Individual raw transcript files (pickle dumps)
│   ├── processed/          # Cleaned, preprocessed, and feature-engineered data
│   │   └── processed_content_data.csv
│   └── models/             # Trained machine learning and NLP models
│       ├── lda_model.pkl
│       ├── lda_model_dict.pkl
│       ├── tfidf_vectorizer.pkl
│       ├── kmeans_lda.pkl
│       ├── kmeans_tfidf.pkl
│       └── ensemble_classifier.pkl
├── pages/                  # Streamlit multi-page application components
│   ├── 01_Content_Success_Predictor.py
│   ├── 02_Audience_Insights_&_Personalization.py
│   └── 03_Cultural_Trend_Dashboard.py
├── scripts/                # Standalone scripts for data acquisition and initial processing
│   ├── scrape_data.py      # Script to scrape and perform initial cleaning
│   └── preprocess_data.py  # Script for further preprocessing and feature engineering
├── src/                    # Core application source code
│   ├── init.py
│   ├── utils/              # General utility functions
│   │   ├── init.py
│   │   ├── logger.py       # Application-wide logging configuration
│   │   └── helpers.py      # Common helper functions (text cleaning, model/data loading)
│   ├── models/             # Machine learning and NLP model definitions and training logic
│   │   ├── init.py
│   │   ├── nlp_pipeline.py     # Functions for applying trained NLP pipeline on new text input
│   │   ├── nlp_topic_modeling.py # Script for training LDA topic model
│   │   ├── clustering.py       # Script for training TF-IDF & KMeans clustering models
│   │   └── prediction_model.py   # Script for training the ensemble classification model
│   └── config.py           # Centralized application configurations and paths
├── app.py                  # Main Streamlit application entry point
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation
├── .gitignore              # Specifies intentionally untracked files to ignore
└── .env                    # Environment variables (e.g., API keys - not committed
```
## Getting Started

### Installation

1.  **Clone the repository:**
```bash
git clone https://github.com/santoshkkashyap25/transnlp.git
cd transnlp
```

2.  **Create and activate a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`


    ```

3.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt


    ```

4.  **Download NLTK and spaCy data:**
    The `src/utils/helpers.py` script will attempt to download necessary NLTK data automatically on its first run if missing.
    For spaCy, download the English model:

    ```bash
    python -m spacy download en_core_web_sm


    ```

### Initial Setup (Data & Models)

For the application to function correctly, you need to first scrape data and train the NLP and prediction models. This process involves executing specific scripts:

1.  **Run Data Scraping & Preprocessing:**

    ```bash
    python scripts/scrape_data.py
    python scripts/preprocess_data.py


    ```

    *These scripts will populate the `data/raw/` and `data/processed/` directories with the necessary CSV files.*

2.  **Train NLP Models:**

    ```bash
    python src/models/nlp_topic_modeling.py
    python src/models/clustering.py


    ```

    *This will train and save the LDA model and its dictionary into `data/models/`.*

3.  **Train Prediction Model:**

    ```bash
    python src/models/prediction_model.py


    ```

    *This will train and save the `ensemble_classifier.pkl` model into `data/models/`.*

### Run the Streamlit App

```bash
streamlit run app.py

```