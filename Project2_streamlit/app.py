import joblib
import streamlit as st
import numpy as np
import re 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Charger les modèles et les outils nécessaires
rf_model = joblib.load('sentiment_rf_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Définir la fonction pour le modèle Random Forest
def predict_sentiment_rf(review):
    review_vector = tfidf_vectorizer.transform([review])
    prediction = rf_model.predict(review_vector)
    return prediction[0]

# Charger le modèle Keras (assurez-vous qu'il est sauvegardé correctement)
# model_keras = ... 

# Chargement du modèle et du tokenizer
model = load_model('model_keras.h5')
tokenizer = joblib.load('tokenizer.pkl')

def preprocess_review(review):
    # Prétraitement de l'avis
    review = review.lower()
    review = re.sub(r'[^\w\s]', '', review)

    # Tokenisation et padding
    sequence = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequence, maxlen=128)  # Assurez-vous que maxlen correspond à votre configuration d'entraînement
    return padded

def predict_sentiment_keras(review):
    review_padded = preprocess_review(review)
    prediction = model.predict(review_padded)
    sentiment_labels = ['negatif', 'neutre', 'positif']
    return sentiment_labels[np.argmax(prediction)]

# Interface Streamlit
st.title('Analyse de Sentiment des Avis d’Assurance')

# Entrée utilisateur
user_review = st.text_area("Entrez votre avis", "")

# Choix du modèle
model_choice = st.selectbox("Choisissez le modèle de prédiction", ["Random Forest", "Keras"])

# Bouton de prédiction
if st.button('Prédire'):
    if model_choice == "Random Forest":
        sentiment = predict_sentiment_rf(user_review)
    else:  # Keras
        sentiment = predict_sentiment_keras(user_review)

    st.write(f"Le sentiment prédit est : {sentiment}")
