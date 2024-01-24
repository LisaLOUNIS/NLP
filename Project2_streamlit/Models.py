import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime as dt
import os
import joblib
import streamlit as st
import string
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
 
 
# Read the CSV file into a Pandas DataFrame
all_data = pd.read_csv("data_scrapped_with_company.csv")
nan_per_column = all_data.isnull().sum()
print(nan_per_column)
data_avis_cor_en_not_null = all_data[all_data['review_en'].notnull()]
duplicate_rows = all_data[all_data.duplicated(keep=False)]
# Trier par valeurs pour regrouper les duplicatas ensemble
duplicate_rows_sorted = duplicate_rows.sort_values(by=list(all_data.columns))
# Afficher les lignes dupliquées avec leurs premières occurrences
print(duplicate_rows_sorted)
#On supprime les doublons
all_data = all_data.drop_duplicates()
 
 
 
# Télécharger les packages nécessaires de nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
 
 
 
def generate_trigrams(token_list):
    return [' '.join(trigram) for trigram in ngrams(token_list, 3) if len(trigram) == 3]
 
def preprocess_text(df, text_column):
    # Check for NaN values in the text column and replace them with empty strings
    df[text_column] = df[text_column].fillna("").astype(str)
 
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
 
    # Création des trigrammes
    df['trigrammes'] = df['review_en_lemmatized'].apply(generate_trigrams)
 
    return df
 
 
all_data = preprocess_text(all_data, 'review_en')
 
 
 
# Joindre les trigrammes en une seule chaîne de caractères pour chaque avis
all_data['trigrammes_joined'] = all_data['trigrammes'].apply(lambda x: ' '.join(x))
 
# Créer un objet TfidfVectorizer
vectorizer = TfidfVectorizer()
 
# Appliquer TF-IDF aux trigrammes joints
tfidf_matrix = vectorizer.fit_transform(all_data['trigrammes_joined'])
 
nltk.download('vader_lexicon')
 
# Fonction de normalisation des ratings
def normalize_rating(rating):
    return (rating - 3) / 2  # Ajustez en fonction de votre échelle de rating
 
# Fonction de combinaison des scores
def combine_scores(df, vader_column, rating_column, method='average'):
    # Normaliser les ratings
    df['normalized_rating'] = df[rating_column].apply(normalize_rating)
 
    if method == 'average':
        df['combined_score'] = (df[vader_column] + df['normalized_rating']) / 2
    elif method == 'weighted':
        df['combined_score'] = 0.6 * df[vader_column] + 0.4 * df['normalized_rating']
    else:
        raise ValueError("Méthode non reconnue")
 
    return df
 
# Exemple d'utilisation
sia = SentimentIntensityAnalyzer()
 
# Calculer le score VADER pour chaque avis
all_data['vader_score'] = all_data['review_en_no_stopwords'].apply(lambda x: sia.polarity_scores(x)['compound'])
 
# Combiner les scores VADER et les ratings
all_data = combine_scores(all_data, 'vader_score', 'stars', method='weighted')
 
# Fonction pour convertir le score en label de sentiment
def label_from_score(score):
    if score > 0.3:
        return 'positif'
    elif score < 0.22:
        return 'négatif'
    else:
        return 'neutre'
 
# Appliquer la fonction pour créer la colonne de labels de sentiment
all_data['sentiment_label'] = all_data['combined_score'].apply(label_from_score)
 
 
 
 
# Conversion du score de sentiment en un DataFrame pour la concaténation
sentiment_scores = all_data['combined_score'].values.reshape(-1, 1)
 
# Concaténation des scores de sentiment avec la matrice TF-IDF
import scipy
 
X_enriched = scipy.sparse.hstack((tfidf_matrix, sentiment_scores))
 
# Séparation des données
X_train, X_test, y_train, y_test = train_test_split(
    X_enriched,
    all_data['sentiment_label'],
    test_size=0.2,
    random_state=42
)
 
# Construction et entraînement du modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
 
# Prédiction et évaluation
y_pred = rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
 
joblib.dump(rf_model, 'sentiment_rf_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
 
#####"keras###
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import re
 

 
# Analyser la taille du vocabulaire
word_counts = sum([len(set(review.split())) for review in all_data['review_en_no_stopwords']])
print(f"Nombre total de mots uniques : {word_counts}")
 
# Calculer la longueur moyenne ou médiane des avis
review_lengths = [len(review.split()) for review in all_data['review_en_no_stopwords']]
average_length = sum(review_lengths) / len(review_lengths)
median_length = sorted(review_lengths)[len(review_lengths) // 2]
print(f"Longueur moyenne des avis : {average_length}")
print(f"Longueur médiane des avis : {median_length}")
 
# Choisissez vocab_size et max_length en fonction de ces valeurs
vocab_size = min(10000, word_counts)
max_length = int(average_length)
 
 
# Création et adaptation du tokenizer
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(all_data['review_en_keras'])
sequences = tokenizer.texts_to_sequences(all_data['review_en_no_stopwords'])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
 
# Padding des séquences pour qu'elles aient toutes la même longueur
data = pad_sequences(sequences, maxlen=max_length)
 
# Conversion des labels en numérique
le = LabelEncoder()
labels = le.fit_transform(all_data['sentiment_label'])
labels = np.asarray(labels)
 
# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
 
# Construction du modèle avec Dropout pour réduire le surapprentissage
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_length))
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))
 
# Compilation du modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
# Callbacks pour l'arrêt précoce et TensorBoard
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
tensorboard_callback = TensorBoard(log_dir='./logs')
 
# Entraînement du modèle avec les callbacks
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
          callbacks=[early_stopping, tensorboard_callback])
 
model.save('keras_model.h5')
joblib.dump(tokenizer, 'tokenizer.pkl')
 