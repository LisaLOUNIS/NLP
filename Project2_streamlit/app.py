import joblib
import streamlit as st
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
 
@st.experimental_memo(ttl=3600, max_entries=10)
def load_models():
    rf_model = joblib.load('sentiment_rf_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = load_model('keras_model.h5')
    tokenizer = joblib.load('tokenizer.pkl')
    return rf_model, tfidf_vectorizer, model, tokenizer
 
 
rf_model, tfidf_vectorizer, keras_model, tokenizer = load_models()
 
# Fonctions pour l'analyse des sentiments
def predict_sentiment_rf(review):
    review_vector = tfidf_vectorizer.transform([review])
    prediction = rf_model.predict(review_vector)
    return prediction[0]
 
def preprocess_review(review):
    review = review.lower()
    review = re.sub(r'[^\w\s]', '', review)
    sequence = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequence, maxlen=67)
    return padded
 
def predict_sentiment_keras(review):
    review_padded = preprocess_review(review)
    prediction = keras_model.predict(review_padded)
    sentiment_labels = ['negatif', 'neutre', 'positif']
    return sentiment_labels[np.argmax(prediction)]
 
 
# Interface Streamlit
st.title('Analyse de Sentiment et Résumé des Avis d’Assurance')
 
# Entrée utilisateur pour l'analyse des sentiments
st.header("Analyse des Sentiments")
user_review = st.text_area("Entrez votre avis pour l'analyse des sentiments", "")
model_choice = st.selectbox("Choisissez le modèle de prédiction", ["Random Forest", "Keras"])
 
 
if st.button('Prédire le Sentiment'):
    if model_choice == "Random Forest":
        sentiment = predict_sentiment_rf(user_review)
    else:  # Keras
        sentiment = predict_sentiment_keras(user_review)
    st.write(f"Le sentiment prédit est : {sentiment}")
 
 