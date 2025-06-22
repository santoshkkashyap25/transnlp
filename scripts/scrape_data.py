# In streamlit_app/scripts/scrape_data.py

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import pickle
import joblib
import os
import re
import string
import imdb
import warnings
from langdetect import detect
import requests 
from src.utils.logger import app_logger
from src.config import SCRAPING_BASE_URL, RAW_DATA_DIR, TRANSCRIPTS_RAW_DIR

warnings.filterwarnings('ignore')

# Initialize IMDb API
ia = imdb.IMDb()

def _fetch_html(url):
    """Helper function to fetch HTML content from a URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        app_logger.error(f"Error fetching URL {url}: {e}")
        return None

def scrape_links(url):
    """Scrapes all content links from the given URL."""
    app_logger.info(f"Scraping content links from {url}")
    html_data = _fetch_html(url)
    if not html_data:
        return []
    soup = BeautifulSoup(html_data, "lxml")
    
    # Try the specific class first (from notebook)
    specific_section = soup.find(class_="elementor-section elementor-top-section elementor-element elementor-element-b70b8d7 elementor-section-boxed elementor-section-height-default elementor-section-height-default")
    if specific_section:
        result = [x.get('href') for x in specific_section.find_all("a")]
    else:
        # Broader search if the specific class isn't found or changes
        app_logger.warning("Specific elementor-section class not found for links, attempting broader search.")
        result = [a.get('href') for a in soup.find_all('a', href=True) if SCRAPING_BASE_URL in a.get('href')]
        result = [link for link in result if "/stand-up-comedy-scripts/" in link and link != SCRAPING_BASE_URL] # Filter to specific content links

    app_logger.info(f"Found {len(result)} links.")
    return result

def scrape_tags(url):
    """Scrapes content titles/tags from the given URL."""
    app_logger.info(f"Scraping tags (titles) from {url}")
    html_data = _fetch_html(url)
    if not html_data:
        return []
    soup = BeautifulSoup(html_data, "lxml")
    
    specific_section = soup.find(class_="elementor-section elementor-top-section elementor-element elementor-element-b70b8d7 elementor-section-boxed elementor-section-height-default elementor-section-height-default")
    if specific_section:
        result = [x.text.strip() for x in specific_section.find_all("h3")]
    else:
        app_logger.warning("Specific elementor-section class not found for tags, attempting broader search.")
        # Attempt to find h3 elements within common content areas
        result = [h3.text.strip() for h3 in soup.find_all('h3')]
        result = [tag for tag in result if tag and ('Stand-Up' in tag or 'Comedy' in tag)] # Basic filtering

    app_logger.info(f"Found {len(result)} tags.")
    return result

def scrape_transcript(url, content_id):
    """
    Scrapes the transcript from a given content URL.
    Checks if the transcript already exists to avoid re-scraping.
    """
    os.makedirs(TRANSCRIPTS_RAW_DIR, exist_ok=True)
    transcript_file_path = os.path.join(TRANSCRIPTS_RAW_DIR, f"{content_id}.pkl")

    if os.path.exists(transcript_file_path):
        app_logger.info(f"Transcript for {content_id} already exists. Skipping scraping.")
        with open(transcript_file_path, "rb") as file:
            return pickle.load(file)

    app_logger.info(f"Scraping transcript from {url} for content ID {content_id}")
    html_data = _fetch_html(url)
    if not html_data:
        app_logger.warning(f"Could not fetch HTML for transcript from {url}")
        return []

    soup = BeautifulSoup(html_data, "lxml")
    
    specific_content_element = soup.find(class_="elementor-element elementor-element-74af9a5b elementor-widget elementor-widget-theme-post-content")
    if specific_content_element:
        result = [x.text for x in specific_content_element.find_all("p")]
    else:
        app_logger.warning("Specific content element class not found for transcript, attempting broader search.")
        # Fallback: look for paragraphs within the main content area (e.g., article body)
        main_content = soup.find('article') or soup.find('main') or soup.find(class_=re.compile(r'content|post|article'))
        if main_content:
            result = [p.text for p in main_content.find_all('p')]
        else:
            app_logger.warning(f"Could not find a suitable content area to extract transcript from {url}.")
            result = []

    if result:
        try:
            with open(transcript_file_path, "wb") as file:
                pickle.dump(result, file)
            app_logger.info(f"Transcript for {content_id} saved to {transcript_file_path}")
        except Exception as e:
            app_logger.error(f"Error saving transcript for {content_id} to file: {e}")
    else:
        app_logger.warning(f"No transcript content found for URL: {url}")
        
    return result

def combine_text(list_of_text):
    """Combines a list of text paragraphs into one large chunk of text."""
    return ' '.join(list_of_text)

def clean_text_content(text):
    """Applies a series of cleaning steps to the text."""
    text = re.sub(r'\\[.*?\\]', '', text)          # Remove text in square brackets
    text = text.lower()                            # Convert text to lowercase
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # Remove punctuation
    text = re.sub('\\n', '', text)                 # Remove newlines
    text = re.sub('[‘’“”…]', '', text)             # Remove specific special characters
    text = re.sub('[♪)(“”…]', '', text)            # Remove additional special characters
    text = re.sub('\\w*\\d\\w*', '', text)           # Remove words containing numbers
    return text


def scrape_imdb_details(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    details = {"title": "Not found", "runtime": "Not found", "rating": "Not found"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # --- Title ---
        title_element = soup.find('h1', {'data-testid': 'hero__pageTitle'})
        if title_element:
            details['title'] = title_element.get_text(strip=True)

        # --- Runtime ---
        for li in soup.find_all('li', class_='ipc-inline-list__item'):
            text = li.get_text(strip=True)
            if re.match(r'^\d+h\s*\d*m?$|^\d+m$', text):
                hours = re.search(r'(\d+)h', text)
                minutes = re.search(r'(\d+)m', text)
                total_minutes = 0
                if hours:
                    total_minutes += int(hours.group(1)) * 60
                if minutes:
                    total_minutes += int(minutes.group(1))
                details['runtime'] = total_minutes  # integer value
                break


        # --- Rating ---
        rating_element = soup.find('span', class_='ipc-rating-star--rating') or \
                         soup.find('span', {'data-testid': 'hero-rating-bar__aggregate-rating__score'})
        if rating_element:
            details['rating'] = rating_element.get_text(strip=True)

    except Exception as e:
        print(f"Scraping error: {e}")

    return details


def get_imdb_info(df):
    from urllib.parse import urljoin

    app_logger.info("Starting IMDb info fetching using direct scraping after IMDbPY search.")
    
    if 'runtime' not in df.columns:
        df['runtime'] = np.nan
    if 'rating' not in df.columns:
        df['rating'] = np.nan

    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    base_imdb_url = "https://www.imdb.com/title/"
    errors_count, successful_fetches = 0, 0

    for i, (index, row) in enumerate(df.iterrows()):
        if pd.notna(row['runtime']) and pd.notna(row['rating']):
            continue

        title = str(row['Tag']).strip()
        if not title:
            df.at[index, 'runtime'] = np.nan
            df.at[index, 'rating'] = np.nan
            errors_count += 1
            continue

        try:
            search_results = ia.search_movie(title[:100])
            if not search_results:
                app_logger.warning(f"No IMDb search results for '{title}'")
                errors_count += 1
                continue

            filtered = [m for m in search_results if 'stand-up' in m.get('kind', '') or 'comedy' in m.get('kind', '')]
            movie_id = (filtered or search_results)[0].movieID
            imdb_url = urljoin(base_imdb_url, f"tt{movie_id}/reference")

            # Direct scraping from IMDb HTML
            scraped = scrape_imdb_details(imdb_url)

            runtime_value = scraped.get('runtime', 'Not found')
            rating_str = scraped.get('rating', 'Not found').strip()

            df.at[index, 'runtime'] = runtime_value if isinstance(runtime_value, int) else np.nan
            df.at[index, 'rating'] = float(rating_str) if re.match(r'^\d+(\.\d+)?$', rating_str) else np.nan
            app_logger.info(f"[Scraped] IMDb info for '{title}': Runtime: {runtime_value}, Rating: {rating_str}")
            successful_fetches += 1

        except Exception as e:
            df.at[index, 'runtime'] = np.nan
            df.at[index, 'rating'] = np.nan
            errors_count += 1
            app_logger.error(f"Error scraping IMDb for '{title}': {e}")

        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            app_logger.info(f"Processed {i+1}/{len(df)} — Success: {successful_fetches}, Errors: {errors_count}")

    app_logger.info(f"IMDb scraping complete. Total rows: {len(df)}, Success: {successful_fetches}, Errors: {errors_count}")
    return df


def scrape_and_clean_data(limit=None):
    app_logger.info(f"Starting data scraping and cleaning process (resumable). Processing limit: {limit if limit is not None else 'None (all)'}")
    
    output_csv_path = os.path.join(RAW_DATA_DIR, 'scraped_and_cleaned_content_data.csv')

    data_frame = pd.DataFrame()
    # Attempt to load existing data to resume
    if os.path.exists(output_csv_path):
        try:
            data_frame = pd.read_csv(output_csv_path)
            app_logger.info(f"Resuming from existing data: {len(data_frame)} records loaded from {output_csv_path}")
            # Ensure proper dtypes for potential NaN columns
            data_frame['runtime'] = pd.to_numeric(data_frame.get('runtime', pd.Series()), errors='coerce')
            data_frame['rating'] = pd.to_numeric(data_frame.get('rating', pd.Series()), errors='coerce')
            data_frame['S No.'] = pd.to_numeric(data_frame.get('S No.', pd.Series()), errors='coerce')
            
            # If resuming with a limit, truncate the loaded data to the limit
            if limit is not None and len(data_frame) > limit:
                data_frame = data_frame.head(limit).copy()
                app_logger.info(f"Truncated loaded data to {limit} entries due to limit parameter.")

        except Exception as e:
            app_logger.warning(f"Could not load existing data from {output_csv_path} for resumption: {e}. Starting fresh.")
            data_frame = pd.DataFrame() # Start fresh if loading fails

    # --- Step 1: Scrape links and tags if not already done or if starting fresh ---
    if data_frame.empty or 'URL' not in data_frame.columns or 'Tag' not in data_frame.columns:
        links = scrape_links(SCRAPING_BASE_URL)
        tags = scrape_tags(SCRAPING_BASE_URL)

        if not links or not tags:
            app_logger.error("Failed to scrape links or tags. Aborting data processing.")
            return pd.DataFrame() # Return empty DataFrame

        min_len = min(len(links), len(tags))
        app_logger.info(f"Initial scrape found {min_len} potential items.")

        # Apply the limit here
        if limit is not None:
            min_len = min(min_len, limit)
            app_logger.info(f"Processing limited to first {min_len} items based on 'limit' parameter.")
            
        links = links[:min_len]
        tags = tags[:min_len]
        
        data_frame = pd.DataFrame({
            "Tag": tags,
            "URL": links
        })
        data_frame.insert(loc=0, column='S No.', value=np.arange(len(data_frame)))
        # Save after initial link/tag scraping
        data_frame.to_csv(output_csv_path, index=False)
        app_logger.info(f"Initial links/tags saved to: {output_csv_path}")
    else:
        app_logger.info("Links and tags already loaded from previous run or present in loaded data.")
        # Ensure that if we loaded data, and a limit is set, we respect it
        if limit is not None and len(data_frame) > limit:
            data_frame = data_frame.head(limit).copy()
            app_logger.info(f"Truncated existing data to {limit} entries based on 'limit' parameter.")


    # Ensure the raw transcripts directory exists
    os.makedirs(TRANSCRIPTS_RAW_DIR, exist_ok=True)

    # --- Step 2: Scrape/Load Transcripts (resumable) ---
    app_logger.info("Scraping/Loading transcripts (resumable).")
    if 'Raw Transcript' not in data_frame.columns:
        data_frame['Raw Transcript'] = [[]] * len(data_frame) # Initialize with empty lists
    
    for index, row in data_frame.iterrows():
        transcript_file_path = os.path.join(TRANSCRIPTS_RAW_DIR, f"{row['S No.']}.pkl")
        # Only call scrape_transcript if the file doesn't exist or if the Raw Transcript in DF is empty/invalid
        if not os.path.exists(transcript_file_path) or not (isinstance(row.get('Raw Transcript'), list) and len(row.get('Raw Transcript', [])) > 0):
            transcript_content = scrape_transcript(row['URL'], row['S No.'])
            data_frame.at[index, 'Raw Transcript'] = transcript_content
            # Save after each transcript to persist progress. This is verbose but safe for resuming.
            data_frame.to_csv(output_csv_path, index=False) 
        
    initial_rows_after_transcript = len(data_frame)
    data_frame = data_frame[data_frame['Raw Transcript'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    if len(data_frame) < initial_rows_after_transcript:
        app_logger.warning(f"Dropped {initial_rows_after_transcript - len(data_frame)} rows due to empty raw transcripts after full transcript pass.")
    data_frame = data_frame.reset_index(drop=True)
    app_logger.info(f"Transcripts processing complete. Current records: {len(data_frame)}")

    # Combine the list of paragraphs into single strings for each transcript
    # Check if 'Transcript' column is missing or if any values are not strings (or empty)
    if 'Transcript' not in data_frame.columns or data_frame['Transcript'].apply(lambda x: not isinstance(x, str) or x.strip() == '').any():
        app_logger.info("Combining raw transcripts into single strings.")
        data_frame['Transcript'] = data_frame['Raw Transcript'].apply(combine_text)
        data_frame.to_csv(output_csv_path, index=False)
        app_logger.info(f"Raw transcripts combined. Current records: {len(data_frame)}")
    else:
        app_logger.info("Transcripts already combined.")

    # --- Step 3: Clean the Transcripts ---
    app_logger.info("Applying cleaning to transcripts.")
    data_frame['Transcript'] = data_frame['Transcript'].apply(clean_text_content)
    
    initial_rows_after_clean = len(data_frame)
    data_frame = data_frame[data_frame['Transcript'].str.strip() != '']
    if len(data_frame) < initial_rows_after_clean:
        app_logger.warning(f"Dropped {initial_rows_after_clean - len(data_frame)} rows due to empty transcripts after cleaning.")
    data_frame = data_frame.reset_index(drop=True)
    data_frame.to_csv(output_csv_path, index=False)
    app_logger.info(f"Transcripts cleaned. Current records: {len(data_frame)}")

    # --- Step 4: Extract Name, Title, Year from Tag ---
    if 'Names' not in data_frame.columns or data_frame['Names'].isnull().any() or \
       'Title' not in data_frame.columns or data_frame['Title'].isnull().any() or \
       'Year' not in data_frame.columns or data_frame['Year'].isnull().any():

        app_logger.info("Extracting Names, Titles, Years from tags.")

        # Step 1: Clean tag column
        data_frame['CleanTag'] = data_frame['Tag'].str.split('|').str[0].str.strip()

        # Step 2: Extract Year
        data_frame['Year'] = data_frame['CleanTag'].str.extract(r'(\d{4})')

        # Step 3: Extract Name
        def extract_name(tag):
            if ':' in tag:
                return tag.split(':')[0].strip()
            if '’s' in tag:
                return tag.split('’s')[0].strip() + '’s'
            if ',' in tag and '(' in tag:
                return tag.split(',')[-1].split('(')[0].strip()
            return tag.split('(')[0].strip()

        data_frame['Names'] = data_frame['CleanTag'].apply(extract_name)

        # Step 4: Extract Title
        def extract_title(row):
            tag = str(row.get('CleanTag') or '')
            name = str(row.get('Names') or '')
            year = str(row.get('Year') or '')
            title = tag.replace(name, '', 1).replace(year, '', 1)
            return re.sub(r'[():]', '', title).strip(' -')

        data_frame['Title'] = data_frame.apply(extract_title, axis=1)

        # Save cleaned data
        data_frame.to_csv(output_csv_path, index=False)
        app_logger.info(f"Names, Titles, Years extracted. Current records: {len(data_frame)}")

    else:
        app_logger.info("Names, Titles, Years already extracted.")


    # --- Step 5: Get IMDb Info ---
    app_logger.info("Fetching IMDb info (resumable). This might take some time.")
    data_frame = get_imdb_info(data_frame)
    data_frame.to_csv(output_csv_path, index=False)
    app_logger.info(f"IMDb info fetch complete. Current records: {len(data_frame)}")

    # Replace empty strings/None with NaN for consistency before saving
    data_frame = data_frame.replace(r'^\s*$', np.nan, regex=True)

    # --- Step 6: Detect Language ---
    # Re-detect if column missing or any NaNs/empty strings
    if 'language' not in data_frame.columns or data_frame['language'].isnull().any() or (data_frame['language'].astype(str).str.strip() == '').any():
        app_logger.info("Detecting language for transcripts.")
        if 'language' not in data_frame.columns:
            data_frame['language'] = np.nan
        
        for index, row in data_frame.iterrows():
            if pd.isna(row['language']) or str(row['language']).strip() == '':
                try:
                    lang = detect(str(row['Transcript'])[:500]) if str(row['Transcript']).strip() else np.nan
                    data_frame.at[index, 'language'] = lang
                except Exception as e:
                    data_frame.at[index, 'language'] = np.nan
                    app_logger.warning(f"Language detection failed for index {index}: {e}")
        
        data_frame.to_csv(output_csv_path, index=False)
        app_logger.info(f"Language detection complete. Distribution:\n{data_frame.language.value_counts()}")
    else:
        app_logger.info("Language already detected.")

    app_logger.info(f"Final data processing complete. Records processed: {len(data_frame)}")

    return data_frame

if __name__ == "__main__":
    app_logger.info("Running scrape_data.py directly.")
    scraped_df = scrape_and_clean_data()
    app_logger.info(f"Scraping and cleaning finished. Processed {len(scraped_df)} records.")