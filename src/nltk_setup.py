import os
import nltk
from src.utils.logger import app_logger

def setup_nltk_data():
    nltk_data_dir = os.path.join("/tmp", "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)

    # Ensure NLTK uses this directory
    os.environ["NLTK_DATA"] = nltk_data_dir
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    # Resources to verify and download if missing
    required_packages = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng"
    }

    app_logger.info(f"NLTK_DATA set to: {nltk_data_dir}")

    for package, test_path in required_packages.items():
        try:
            nltk.data.find(test_path)
            app_logger.info(f"NLTK resource found: {package}")
        except LookupError:
            app_logger.info(f"⬇ Downloading missing NLTK resource: {package}")
            try:
                nltk.download(package, download_dir=nltk_data_dir)
                app_logger.info(f"Downloaded: {package}")
            except Exception as e:
                app_logger.error(f"Failed to download {package}: {e}", exc_info=True)

# Run setup immediately on import
setup_nltk_data()
