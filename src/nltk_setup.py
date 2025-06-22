import nltk
import os
import sys

def setup_nltk_data():
    nltk_data_dir = os.path.join("/tmp", "nltk_data")

    if not os.path.exists(nltk_data_dir):
        os.makedirs(nltk_data_dir)

    os.environ["NLTK_DATA"] = nltk_data_dir

    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    required_packages = [
        "punkt",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger",
        "punkt_tab",
         "averaged_perceptron_tagger_eng"
    ]

    print(f"NLTK_DATA set to: {nltk_data_dir}") # Log for debugging

    for package in required_packages:
        try:
            # Check if the package is already available
            nltk.data.find(f'{package}')
            print(f"NLTK package '{package}' already present.")
        except LookupError:
            # If not found, download it
            print(f"Downloading NLTK package: {package} to {nltk_data_dir}...")
            try:
                nltk.download(package, download_dir=nltk_data_dir)
                print(f"Successfully downloaded {package}.")
            except Exception as e:
                print(f"Error downloading {package}: {e}")
                # You might want more robust error handling here depending on your needs

# Call this setup function immediately when this module is imported
setup_nltk_data()
