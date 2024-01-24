import streamlit as st
import pandas as pd
from nltk.stem import WordNetLemmatizer
import nltk
from nltk import FreqDist
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords') 
nltk.download('wordnet')
nltk.download('punkt')
import string
import os

import matplotlib.pyplot as plt
import seaborn as sns

def generate_trigrams(token_list):
    return [' '.join(trigram) for trigram in ngrams(token_list, 3) if len(trigram) == 3]

def preprocess_text(df, text_column):
    # Vérification des valeurs manquantes et remplacement par string vides
    df[text_column] = df[text_column].fillna("").astype(str)

    # Nettoyage de base du texte
    df[text_column] = df[text_column].str.lower().str.translate(str.maketrans('', '', string.punctuation))

    # Tokenisation
    df['review_en_tokenized'] = df[text_column].apply(word_tokenize)

    # Suppression des stop words
    stop_words = set(stopwords.words('english'))
    symbols = ['“', '”', 'malakoff', 'macif', 'lcl', 'matmut', 'allianz', 'axa', 'mutex', 'swiss', 'generali', 'gmf']
    df['review_en_no_stopwords'] = df['review_en_tokenized'].apply(lambda x: [word for word in x 
                                                                              if word not in stop_words 
                                                                              and word not in symbols and len(word)>2])

    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    df['review_en_lemmatized'] = df['review_en_no_stopwords'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    # Création des trigrammes
    df['trigrammes'] = df['review_en_lemmatized'].apply(generate_trigrams)

    return df


@st.cache_data
def load_data():
    all_data = pd.read_csv('data_scrapped_with_company.csv', nrows=3000)
    all_data = 
    all_data['company'] = all_data['company'].str.replace(r'^https://www.opinion-assurances.fr/assureur-', '', regex=True)
    all_data['company'] = all_data['company'].str.replace(r'\.html.*$', '', regex=True)
    all_data['company'] = all_data['company'].str.replace('-', ' ', regex=True)
    all_data = all_data.drop_duplicates()
    all_data = all_data.dropna(subset=('review'))
    all_data = preprocess_text(all_data, 'review_en')

    return all_data

all_data = load_data()

st.title('Statistiques du dataset')

# Affichez les premières lignes de l'ensemble de données
st.subheader('Aperçu des données')
st.write(all_data.head())

# Affichez les informations de l'ensemble de données
st.subheader('Informations sur les données')
st.write(all_data.info())

# Affichez les statistiques descriptives de l'ensemble de données
st.subheader('Statistiques descriptives')
st.write(all_data.describe())

# Créez un histogramme pour chaque colonne numérique
# st.subheader('Histogrammes')
# numeric_columns = all_data.select_dtypes(['int64', 'float64']).columns
# for col in numeric_columns:
#     fig, ax = plt.subplots()
#     all_data[col].hist(ax=ax)
#     st.pyplot(fig)