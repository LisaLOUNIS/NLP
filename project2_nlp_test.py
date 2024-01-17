import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime as dt
import os 

# Chemin vers le dossier contenant les fichiers Excel
folder_path = r'C:\Users\lisac\Downloads\Traduction avis clients\Traduction avis clients'
 
# Liste pour stocker toutes les DataFrames
all_dataframes = []
 
# Parcourir tous les fichiers dans le dossier
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_excel(file_path)
        all_dataframes.append(df)
# Concaténer toutes les DataFrames en une seule
all_data = pd.concat(all_dataframes, ignore_index=True)
nan_per_column = all_data.isnull().sum()
print(nan_per_column)
data_avis_cor_en_not_null = all_data[all_data['avis_cor_en'].notnull()]
print(data_avis_cor_en_not_null[['avis_en','avis_cor_en']])
# Remplacer 'avis_en' par 'avis_cor_en' lorsque 'avis_cor_en' n'est pas null
all_data['avis_en'] = all_data['avis_cor_en'].combine_first(all_data['avis_en'])
all_data['avis'] = all_data['avis_cor'].combine_first(all_data['avis'])
all_data.drop(['avis_cor', 'avis_cor_en'], axis=1, inplace=True)
all_data= all_data.dropna(subset=['avis_en'])
print(all_data.columns)
# Identifier les lignes dupliquées
duplicate_rows = all_data[all_data.duplicated(keep=False)]
# Trier par valeurs pour regrouper les duplicatas ensemble
duplicate_rows_sorted = duplicate_rows.sort_values(by=list(all_data.columns))
# Afficher les lignes dupliquées avec leurs premières occurrences
print(duplicate_rows_sorted)
#On supprime les doublons
all_data = all_data.drop_duplicates()

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
    df['avis_en_tokenized'] = df[text_column].apply(word_tokenize)

    # Suppression des stop words
    stop_words = set(stopwords.words('english'))
    df['avis_en_no_stopwords'] = df['avis_en_tokenized'].apply(lambda x: [word for word in x if word not in stop_words])

    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    df['avis_en_lemmatized'] = df['avis_en_no_stopwords'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    return df

# Appliquer la fonction à votre DataFrame
from collections import Counter
all_data = preprocess_text(df_reviews, 'avis_en')
# Après avoir appliqué preprocess_text
all_words = [word for tokens in all_data['avis_en_lemmatized'] for word in tokens]
word_freq = Counter(all_words)
most_common_words = word_freq.most_common(10)

print(most_common_words)

from nltk.util import ngrams

def extract_ngrams_from_tokenized_data(tokenized_data, num):
    # Générer des n-grams à partir des listes de mots tokenisés
    all_ngrams = [ngram for tokens in tokenized_data for ngram in ngrams(tokens, num)]
    return all_ngrams

# Appliquer la fonction de prétraitement
all_data = preprocess_text(all_data, 'avis_en')

# Générer des bigrammes à partir des données tokenisées et lemmatisées
bigrams = extract_ngrams_from_tokenized_data(all_data['avis_en_lemmatized'], 2)

# Compter la fréquence des bigrammes
bigram_counts = Counter(bigrams)
most_common_bigrams = bigram_counts.most_common(10)  # top 10 bigrammes
print(most_common_bigrams)

# Générer des trigrammes à partir des données tokenisées et lemmatisées
trigrams = extract_ngrams_from_tokenized_data(all_data['avis_en_lemmatized'], 3)

# Compter la fréquence des trigrammes
trigram_counts = Counter(trigrams)
most_common_trigrams = trigram_counts.most_common(10) 
print(most_common_trigrams)

