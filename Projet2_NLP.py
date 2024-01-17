import requests
from bs4 import BeautifulSoup
import pandas as pd
import concurrent.futures
import re  # Regular expression library

def scrape_page(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to retrieve the webpage: {url}")
            return []
    except Exception as e:
        print(f"Error occurred while fetching the webpage: {url}, Error: {e}")
        return []
 
    soup = BeautifulSoup(response.content, 'html.parser')
    reviews = []
 
 
    for review_block in soup.find_all("div", class_="oa_reactionBlock"):
        review = {}
 
        company_name = review_block.find("span", class_="oa_obflink")
        review['company'] = company_name.text.strip() if company_name else "Unknown Company"
 
        # Extracting stars
        stars = review_block.find("div", class_="oa_stackLevel")
        review['stars'] = len(stars.find_all("i", class_="fas fa-star active")) if stars else "Pas noté"
 
        # Extracting author name
        author_name = review_block.find("span", itemprop="author")
        review['author'] = author_name.text if author_name else "No author"
 
        # Extracting comment date and formatting it using regular expression
        comment_date = review_block.find("div", class_="oa_date")
        if comment_date:
            date_str = comment_date.text.strip()
            match = re.search(r'\d{2}/\d{2}/\d{4}', date_str)
            review['date'] = match.group(0) if match else "No date"
        else:
            review['date'] = "No date"
 
        # Extracting review text
        review_text = review_block.find("h4", class_="oa_reactionText")
        review['review'] = review_text.text.strip() if review_text else "No text"
 
        reviews.append(review)
 
    return reviews

def create_and_scrape_url(base_url, page_number):
    if "?page=" in base_url:
        url = base_url + str(page_number)
    else:
        url = base_url + "?page=" + str(page_number) if page_number > 1 else base_url
    return scrape_page(url)

# Base URL without the page number
base_urls = ['https://www.opinion-assurances.fr/assureur-abeille-assurances.html?page=','https://www.opinion-assurances.fr/assureur-caisse-d-epargne.html','https://www.opinion-assurances.fr/assureur-axa.html', 'https://www.opinion-assurances.fr/assureur-allianz.html','https://www.opinion-assurances.fr/assureur-credit-mutuel.html','https://www.opinion-assurances.fr/assureur-direct-assurance.html','https://www.opinion-assurances.fr/assureur-matmut.html','https://www.opinion-assurances.fr/assureur-cnp-assurances.html','https://www.opinion-assurances.fr/assureur-generali.html','https://www.opinion-assurances.fr/assureur-harmonies-mutuelles.html','https://www.opinion-assurances.fr/assureur-mutex.html','https://www.opinion-assurances.fr/assureur-macif.html','https://www.opinion-assurances.fr/assureur-lcl.html','https://www.opinion-assurances.fr/assureur-gmf.html','https://www.opinion-assurances.fr/assureur-cic.html','https://www.opinion-assurances.fr/assureur-olivier-assurances.html', 'https://www.opinion-assurances.fr/assureur-societe-generale-assurances.html', 'https://www.opinion-assurances.fr/assureur-swiss-life.html', 'https://www.opinion-assurances.fr/assureur-malakoff-humanis.html', 'https://www.opinion-assurances.fr/assureur-gmf.html']

# Define the number of pages you want to scrape
number_of_pages = 40  # Adjust as needed

all_reviews = []

# Use ThreadPoolExecutor for parallel scraping
# Use ThreadPoolExecutor for parallel scraping
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    # Prepare a dictionary to hold the futures and corresponding URL info
    future_to_url = {}

    # Loop over each base URL and each page number
    for base_url in base_urls:
        for page_number in range(1, number_of_pages + 1):
            # Submit a new task to the executor: scraping a specific URL
            future = executor.submit(create_and_scrape_url, base_url, page_number)
            # Store this future in the dictionary with its corresponding URL info
            future_to_url[future] = (base_url, page_number)

    # Process the results of the futures as they complete
    for future in concurrent.futures.as_completed(future_to_url):
        # Retrieve the URL info corresponding to this future
        url_info = future_to_url[future]
        try:
            # Get the result of the future (the return value of the scraping function)
            data = future.result()
            # Extend the all_reviews list with the scraped data
            all_reviews.extend(data)
        except Exception as exc:
            # If an error occurred, print the URL info and the error message
            print('%r generated an exception: %s' % (url_info, exc))

# Creating a DataFrame from the accumulated reviews
df_reviews = pd.DataFrame(all_reviews)

from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='fr', target='en')

def translate_text_segmented(text, translator, max_length=4900):
    # Ensure text is a string
    text = str(text)
    # Split the text into segments, ensuring each is within the character limit
    segments = []
    while text:
        if len(text) > max_length:
            # Find nearest space to avoid cutting words
            split_index = text.rfind(' ', 0, max_length)
            if split_index == -1:  # No space found, force split
                split_index = max_length
            segments.append(text[:split_index])
            text = text[split_index:].lstrip()  # Remove leading whitespace for the next segment
        else:
            segments.append(text)
            break
    # Translate each segment
    translated_segments = [translator.translate(segment) for segment in segments]
    # Combine the translated segments
    return ' '.join(translated_segments)

# Apply the segmented translation function to the 'text' column
df_reviews['review_en'] = df_reviews['review'].apply(lambda x: translate_text_segmented(x, translator))

# Display the DataFrame with the translated reviews
print(df_reviews)

df_reviews.to_csv('data_scrapped.csv', index=False)


import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Télécharger les packages nécessaires de nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(df, text_column):
    # Nettoyage de base du texte
    df[text_column] = df[text_column].str.lower().str.translate(str.maketrans('', '', string.punctuation))

    # Tokenisation
    df['review_en_tokenized'] = df[text_column].apply(word_tokenize)

    # Suppression des stop words
    stop_words = set(stopwords.words('english'))
    df['review_en_no_stopwords'] = df['review_en_tokenized'].apply(lambda x: [word for word in x if word not in stop_words])

    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    df['review_en_lemmatized'] = df['review_en_no_stopwords'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    return df






# # Appliquer la fonction à votre DataFrame
# from collections import Counter
# all_data = preprocess_text(df_reviews, 'review_en')
# # Après avoir appliqué preprocess_text
# all_words = [word for tokens in all_data['review_en_lemmatized'] for word in tokens]
# word_freq = Counter(all_words)
# most_common_words = word_freq.most_common(10)

# print(most_common_words)

# from nltk.util import ngrams

# def extract_ngrams_from_tokenized_data(tokenized_data, num):
#     # Générer des n-grams à partir des listes de mots tokenisés
#     all_ngrams = [ngram for tokens in tokenized_data for ngram in ngrams(tokens, num)]
#     return all_ngrams

# # Appliquer la fonction de prétraitement
# all_data = preprocess_text(all_data, 'review_en')

# # Générer des bigrammes à partir des données tokenisées et lemmatisées
# bigrams = extract_ngrams_from_tokenized_data(all_data['review_en_lemmatized'], 2)

# # Compter la fréquence des bigrammes
# bigram_counts = Counter(bigrams)
# most_common_bigrams = bigram_counts.most_common(10)  # top 10 bigrammes
# print(most_common_bigrams)

# # Générer des trigrammes à partir des données tokenisées et lemmatisées
# trigrams = extract_ngrams_from_tokenized_data(all_data['review_en_lemmatized'], 3)

# # Compter la fréquence des trigrammes
# trigram_counts = Counter(trigrams)
# most_common_trigrams = trigram_counts.most_common(10) 
# print(most_common_trigrams)







