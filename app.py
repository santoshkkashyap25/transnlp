import streamlit as st

import pickle
import joblib

import math
import random
import numpy as np
import pandas as pd
import spacy

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#from wordcloud import WordCloud

from PIL import Image
from collections import defaultdict
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_colwidth', 2000)
sns.set(color_codes=True)

def main():
    
    nlp_icon = Image.open("NLP.jpg")
    icon = nlp_icon.resize((600, 400))
    st.set_page_config( page_title="Real Time Transcripts Analysis", page_icon=icon, layout="wide", initial_sidebar_state="collapsed")
    
       # Load data function
    def Load_data():
        df = pd.read_csv("frame.csv")
        return df
    
    # defines homepage
    def home():
        st.title("Real Time Transcripts Analysis")
        st.markdown("### Machine Learning & Natural Language Processing Project")
        st.write("Please select a page on the left sidebar")

        # Centered image with rounded corners
        st.image(icon, caption="TransNLP", use_column_width=False, output_format="PNG")

        # Add some space between sections
        st.markdown("<br><br>", unsafe_allow_html=True)

        # Create a three-column layout
        col1, col2, col3 = st.columns(3)

        # Column 1
        with col1:
            st.header("About Project")
            st.write("""
                Natural language processing (NLP) is an exciting branch of artificial intelligence (AI) 
                that allows machines to break down and understand human language. This project includes 
                text pre-processing techniques, machine learning techniques, and Python libraries for NLP.
            """)

        # Column 2
        with col2:
            st.header("Project Flow")
            st.write("""
                1. **Web Scraping:** Scrape transcripts from an HTML webpage.
                2. **Text Pre-processing:** Use wordcloud, textblob, and gensim for cleaning and tokenization.
                3. **Data Exploration:** Discover common words using word clouds.
                4. **Sentiment Analysis:** Analyze sentiment with TextBlob (polarity and subjectivity).
                5. **Topic Modelling:** Use LDA to find various topics in the corpus.
                6. **Text Generation:** Utilize Markov chains for text generation.
                7. **Rating Prediction:** Train an ensemble classifier using LDA vector and "rating_type" feature.
            """)

        # Column 3
        with col3:
            st.header("Contact Us")
            st.write("Feel free to reach out if you have any questions or suggestions!")
            st.info("Email: santoshkkashyap25@gmail.com")

        # Add a separator
        st.markdown("<hr>", unsafe_allow_html=True)

        # Footer
        st.write("© 2023 Real Time Transcripts Analysis. All rights reserved.")
        

    # Showing the cleaned Dataset
    def show_data():
        df = Load_data()
        data = st.selectbox(
            "Select an option to view",
            [
                "Show Full Dataset",
                "Show Dataset Head",
                "Show Dataset Tail",
                "Show Comedian Names",
                "Show Dataset Description"
            ],
        )

        st.subheader("Explore Dataset")

        if data == "Show Full Dataset":
            st.dataframe(df.style.highlight_max(axis=0), height=500)

        elif data == "Show Dataset Head":
            st.dataframe(df.head().style.highlight_max(axis=0))

        elif data == "Show Dataset Tail":
            st.dataframe(df.tail().style.highlight_max(axis=0))

        elif data == "Show Comedian Names":
            st.write("Unique Comedian Names:")
            st.write(df['Names'].unique())

        elif data == "Show Dataset Description":
            st.table(df.describe().style.highlight_max(axis=0))
            
    # Transcripts function with additional details
    def transcripts():
        st.write("Scraped The Web.")
        st.markdown("### Explore Transcripts")
        st.text("Please Enter Serial No. or Search by Comedian Name")

        df = Load_data()

        if st.button("SHOW DATASET"):
            st.dataframe(df)

        load_transcripts = joblib.load('transcripts_joblib')

        search_option = st.radio(
            "Select Search Option",
            ["By Serial No.", "By Comedian Name"]
        )

        if search_option == "By Serial No.":
            num = st.number_input('Enter Serial No.', min_value=0, max_value=len(df)-1, value=0)
            num = int(num)
        else:
            comedian_name = st.text_input("Enter Comedian Name", "")
            matching_indices = df[df['Names'].str.contains(comedian_name, case=False)].index.tolist()
            if matching_indices:
                num = st.selectbox("Select Serial No.", matching_indices)
            else:
                st.warning("No matching comedian found. Please try again.")
                return

        st.subheader(f"Transcript Details for Serial No. {num}")
        st.write("URL:", df['URL'][num])
       
        st.write("Comedian Name:", df['Names'][num])
        st.write("Title:", df['Title'][num])
        st.write("Year:", df['Year'][num])
        st.write("Runtime:", df['runtime'][num])
        #st.write("Rating:", df['rating'][num])
        st.write("Language:", df['language'][num])

        st.subheader("Transcript")
        st.text_area("Transcript", load_transcripts[num], height=300, max_chars=None, key=None)


    def kde():
        frame = pd.read_csv("frame2.csv")

        plot = st.selectbox(
            "Select a Plot for Visualization",
            [
                "Transcript Character Count KDE",
                "Runtime KDE",
                "IMDb Rating KDE",
                "F-Words Count KDE",
                "S-Words Count KDE",
                "Word Diversity KDE",
                "Diversity / Total words KDE"
            ],
        )

        if plot == "Transcript Character Count KDE":
            x = [len(x) for x in frame.Transcript]
            fig = plt.figure()
            sns.kdeplot(x, shade=True, color="b")
            plt.title('Transcript Character Count KDE')
            st.pyplot(fig)
            mean = np.array(x).mean()
            sd = np.array(x).std()
            st.write(f'Mean: {mean}')
            st.write(f'Standard Deviation: {sd}')

        if plot == "Runtime KDE":
            x = [int(i) for i in frame.runtime if i > 0]
            fig = plt.figure()
            sns.kdeplot(x, shade=True, color="r")
            plt.title('Runtime KDE')
            plt.xlabel('Minutes')
            st.pyplot(fig)
            mean = np.array(x).mean()
            sd = np.array(x).std()
            st.write(f'Mean: {mean}')
            st.write(f'Standard Deviation: {sd}')

        if plot == "IMDb Rating KDE":
            x = [i for i in frame.rating if i > 0]
            fig = plt.figure()
            sns.kdeplot(x, shade=True, color="g")
            plt.title('IMDb Rating KDE')
            st.pyplot(fig)
            mean = np.array(x).mean()
            sd = np.array(x).std()
            st.write(f'Mean: {mean}')
            st.write(f'Standard Deviation: {sd}')

        if plot == "F-Words Count KDE":
            fig = plt.figure()
            sns.kdeplot(frame.f_words, shade=True, color="r")
            plt.title('F-Words Count KDE')
            st.pyplot(fig)
            mean = frame.f_words.mean()
            sd = frame.f_words.std()
            st.write(f'Mean: {mean}')
            st.write(f'Standard Deviation: {sd}')

        if plot == "S-Words Count KDE":
            fig = plt.figure()
            sns.kdeplot(frame.s_words, shade=True, color="r")
            plt.title('S-Words Count KDE')
            st.pyplot(fig)
            mean = frame.s_words.mean()
            sd = frame.s_words.std()
            st.write(f'Mean: {mean}')
            st.write(f'Standard Deviation: {sd}')

        if plot == "Word Diversity KDE":
            fig = plt.figure()
            sns.kdeplot(frame.diversity, shade=True, color="purple")
            plt.title('Word Diversity KDE')
            st.pyplot(fig)
            mean = frame.diversity.mean()
            sd = frame.diversity.std()
            st.write(f'Mean: {mean}')
            st.write(f'Standard Deviation: {sd}')

        if plot == "Diversity / Total words KDE":
            fig = plt.figure()
            sns.kdeplot(frame.diversity_ratio, shade=True, color="g")
            plt.title('Diversity / Total words KDE')
            st.pyplot(fig)
            mean = frame.diversity_ratio.mean()
            sd = frame.diversity_ratio.std()
            st.write(f'Mean: {mean}')
            st.write(f'Standard Deviation: {sd}')

    def rating():
        frame = pd.read_csv("frame2.csv")

        st.write("Given a 1 for any rating above the mean, and a 0 otherwise. This will be our target for a classification task.")
        st.write("High rating (> mean) And Low rating (< mean)")

        frame['rating_type'] = frame.rating.apply(lambda x: 1 if x >= frame.rating.mean() else 0)
        title = 'Counts of specials with higher or lower than average ratings'

        fig = plt.figure()
        sns.countplot(x='rating_type', data=frame)
        plt.title("Counts of specials with higher or lower than average ratings")
        st.pyplot(fig)

    def pair():
        frame = pd.read_csv("frame2.csv")
        st.write("Pairplot Visualization To Discover Correlations")
        fig = sns.pairplot(frame[['diversity_ratio', 'diversity', 'word_count', 'runtime', 'rating', 'rating_type']])
        st.pyplot(fig)


    # def wordcloud():
    #     df = Load_data()
    #     st.markdown("### Enter the Serial No. of the Transcript to see it's Wordcloud.")
    #     st.text("Please check Serial No. here")
    #     if st.button("SHOW DATASET"):
    #         st.dataframe(df)
    #     num = st.number_input('Enter Serial Number', min_value=0, max_value=len(df)-1, value=0)
    #     num = int(num)
    #     st.write((df.title[num]))
    #     wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='midnightblue')
    #     wordcloud.generate(' '.join(df.words[num]))
    #     wordcloud.to_image()

    # for the sentiment Analysis
    
    def sentiments():
        senti = st.selectbox(
            "Select an Option to Explore",
            [
                "Show Individual Sentiments",
                "Show Sentiment Plot",
                "Show Sentiment Plot Over Routine",
            ],
        )
        if senti == "Show Individual Sentiments":
            df = Load_data()
            st.subheader("Choose an option to analyze sentiments:")

            option = st.radio("Select Option", ["Serial Number", "Comedian Name"])

            if option == "Serial Number":
                st.text("Please Check Serial No. Here")
                if st.button("SHOW DATASET"):
                    st.dataframe(df)

                num = st.number_input('Enter Serial Number', min_value=0, max_value=len(df)-1, value=0)
                num = int(num)

                i = df.loc[num]

            elif option == "Comedian Name":
                if st.button("Show Comedian Names"):
                    st.write(df['Names'].unique())
                text = st.text_input('Enter Comedian Name')
                matching_names = df[df["Names"].str.lower().str.contains(text.lower())]['Names'].unique()

                if not matching_names.any():
                    st.warning("No matching names found. Please try again.")
                    st.stop()

                selected_name = st.selectbox('Select Comedian Name', matching_names)

                filtered_df = df[df["Names"].str.lower() == selected_name.lower()]

                if not filtered_df.empty:
                    i = filtered_df.iloc[0]  # Take the first row if there are multiple matches
                else:
                    st.warning("No data found for the selected comedian name. Please try again.")
                    st.stop()

            dff = pd.DataFrame()
            dff['S.No'] = [i['S No.']]
            dff['Name'] = [i['Names']]
            dff['Title Name'] = [i['Title']]
            transcript = str(i['Transcript'])

            pol = TextBlob(transcript).sentiment.polarity
            sub = TextBlob(transcript).sentiment.subjectivity

            dff['Polarity'] = [pol]
            dff['Subjectivity'] = [sub]

            st.write("Sentiments Analysis")
            st.table(dff)
            
        if senti == "Show Sentiment Plot":
            df = pd.read_csv("frame2.csv")
            st.write("Explore Sentiments with a Subset of the Dataset.")
            st.markdown("### Enter the serial numbers of the transcripts to visualize the plot.")
            st.text("Check serial numbers here.")

            if st.button("SHOW DATASET"):
                st.dataframe(df)

            num1 = st.number_input('Enter Starting Serial Number', min_value=0, max_value=20, value=0)
            num1 = int(num1)
            num2 = st.number_input('Enter Ending Serial Number', min_value=0, max_value=20, value=5)
            num2 = int(num2)

            if num1 >= len(df) or num2 >= len(df) or num1 > num2:
                st.error("Invalid input. Please enter valid serial numbers within the range of your dataset.")
            else:
                data = df[num1:num2+1]

                st.subheader("Sentiment Analysis Plot")
                fig, ax = plt.subplots(figsize=(12, 8))

                for index, comedian in enumerate(data['S No.']):
                    transcript = data['Transcript'].loc[comedian]
                    pol = TextBlob(transcript).sentiment.polarity
                    sub = TextBlob(transcript).sentiment.subjectivity

                    x = pol
                    y = sub
                    ax.scatter(x, y, color='red', s=100, alpha=0.7)

                    # Display only the first name
                    first_name = data['Names'][index + num1].split()[0]
                    ax.text(x + .001, y + .002, first_name, fontsize=12)  # Adjust the values here

                ax.set_title('Sentiment Analysis Plot', fontsize=20)
                ax.set_xlabel('<-- Negative -------- Positive -->', fontsize=15)
                ax.set_ylabel('<-- Facts -------- Opinions -->', fontsize=15)
                ax.grid(True, linestyle='--', alpha=0.6)

                st.pyplot(fig)

        if senti == "Show Sentiment Plot Over Routine":
            df = pd.read_csv("frame.csv")
            st.write("To get Sentiment Plot over Routine pick a forward Subset of the Dataset")
            st.markdown("### Enter the Serial Numbers of the Transcript to visualize Plot.")
            st.text("Please check Serial Numbers here")
            if st.button("SHOW DATASET"):
                st.dataframe(df)
            num1 = st.number_input('Enter Starting Serial Number', min_value=0, max_value=20, value=0)
            num1 = int(num1)
            num2 = st.number_input('Enter Ending Serial Number', min_value=0, max_value=20, value=4)
            num2 = int(num2)
            data = df[num1:num2+1]
            st.subheader("Sentiment Plot Over Routine")

            def split_text(text, n=10):
                length = len(text)
                size = math.floor(length / n)
                start = np.arange(0, length, size)
                split_list = []
                for piece in range(n):
                    split_list.append(text[start[piece]:start[piece] + size])
                return split_list

            list_pieces = []
            for t in data.Transcript:
                split = split_text(t)
                list_pieces.append(split)

            polarity_transcript = []
            for lp in list_pieces:
                polarity_piece = []
                for p in lp:
                    polarity_piece.append(TextBlob(p).sentiment.polarity)
                polarity_transcript.append(polarity_piece)

            fig = plt.figure()
            plt.rcParams['figure.figsize'] = [25, 25]
            for index, comedian in enumerate(data['S No.']):
                plt.subplot(4, 5, index + 1)
                plt.plot(polarity_transcript[index])
                plt.plot(np.arange(0, 10), np.zeros(10))
                first_name = data['Names'][index + num1].split()[0]
                plt.title(first_name)
                plt.ylim(ymin=-.2, ymax=.3)
            st.pyplot(fig)

    # Load a pre-trained English NLP model
    nlp = spacy.load("en_core_web_sm")

    def markov_chain(text, exclude_words=None, order=1):
        words = text.split(' ')

        if exclude_words:
            words = [word for word in words if word not in exclude_words]

        m_dict = defaultdict(list)

        for i in range(len(words)-order):
            current_state = tuple(words[i:i+order])
            next_state = words[i+order]
            m_dict[current_state].append(next_state)

        m_dict = dict(m_dict)
        return m_dict

    def generate_sentence(chain, count=100, order=1):
        current_state = random.choice(list(chain.keys()))
        sentence = ' '.join(current_state).capitalize()

        for i in range(count-order):
            next_state = random.choice(chain[current_state])
            sentence += ' ' + next_state
            current_state = tuple(list(current_state[1:]) + [next_state])

        sentence += '.'
        return sentence

    def text():
        data = pd.read_csv("frame2.csv")
        st.markdown("### Enter the Serial No. of the Transcript for Text Generation.")
        st.text("Please check Serial No. here -")

        df = Load_data()
        if st.checkbox("SHOW DATASET"):
            st.dataframe(df)

        num = st.number_input('Enter Number', min_value=0, max_value=len(df)-1, value=0)
        num = int(num)

        st.header(df.Title[num])
        com_text = data.Transcript.loc[num]

        # Define words to exclude from the generated text
        exclude_words = data[['f_words', 's_words']].iloc[num].tolist()

        # Use a higher-order Markov model
        com_dict = markov_chain(com_text, exclude_words, order=2)

        st.write("The Generated Text -")
        st.write(generate_sentence(com_dict, order=2))
        
    # defines topic modelling
    def topic():
        dff = pd.read_csv("frame3.csv")
        st.markdown("### Enter the Serial No. of the Transcript.")
        st.text("Please check Serial No. here -")
        df = Load_data()
        if st.checkbox("SHOW DATASET"):
            st.dataframe(df)
        num = st.number_input('Enter Number', min_value=0, max_value=len(df)-1, value=0)
        num = int(num)
        st.write(dff.Title[num])

        # Extract topic probabilities
        topic_probs = dff.loc[num, 'Culture':'Politics']

        # Convert to numeric type and handle missing values
        topic_probs_numeric = pd.to_numeric(topic_probs, errors='coerce')

        # Check if there are any missing values
        if topic_probs_numeric.isna().any():
            st.write("Warning: Topic probabilities contain missing values.")

        # Get the leading topic
        leading_topic = topic_probs_numeric.idxmax()
        leading_prob = topic_probs_numeric.max()

        # Display the leading topic with maximum probability
        st.write(f"Leading Topic: {leading_topic}")
        st.write(f"Probability: {leading_prob:.4f}")

        # Display all topic probabilities with a title
        st.write("Topic Probabilities:")
        st.table(pd.DataFrame({'Topics': topic_probs_numeric.index, 'Probabilities': topic_probs_numeric.values}))


    page = st.sidebar.selectbox(
    "Select a Page",
    ["Homepage", "Dataset", "Transcripts", "Visualization", "Sentiment Analysis", "Text Generation", "Topic Modelling"],
)

    if page == "Homepage":
        home()

    elif page == "Dataset":
        st.markdown("## Real Time Transcripts Analysis")
        if st.checkbox("Explore Dataset"):
            show_data()

    elif page == "Transcripts":
        st.markdown("## Real Time Transcripts Analysis")
        if st.checkbox("Show Transcripts"):
            transcripts()

    elif page == "Visualization":
        st.markdown("## Real Time Transcripts Analysis")
        st.write("KDE plots ensure everything looks right. A quick view of a simple distribution can be a great indicator.")
        if st.checkbox("Show KDE Plots"):
            kde()
        elif st.checkbox("Show Rating Type Count Plot"):
            rating()
        elif st.checkbox("Show Pairplot Visualization"):
            pair()

    elif page == "Sentiment Analysis":
        st.markdown("## Real Time Transcripts Analysis")
        st.write("A corpus' sentiment is the average of Polarity and Subjectivity.")
        st.write("Polarity: How positive or negative a word is. -1 is very negative. +1 is very positive.")
        st.write("Subjectivity: How subjective or opinionated a word is. 0 is fact. +1 is very much an opinion.")
        sentiments()

    elif page == "Text Generation":
        st.markdown("## Real Time Transcripts Analysis")
        text()

    elif page == "Topic Modelling":
        st.markdown("## Real Time Transcripts Analysis")
        topic()
    
if __name__ == '__main__':
    main()
