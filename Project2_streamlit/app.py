import joblib
import streamlit as st
# Charger le modèle et le vectorizer
model = joblib.load('sentiment_rf_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_sentiment(review):
    # Vectoriser l'avis
    review_vector = vectorizer.transform([review])
    # Prédire le sentiment
    prediction = model.predict(review_vector)
    return prediction[0]

st.title('Analyse de Sentiment des Avis d’Assurance avec TFIDF et Random Forest Classifier')

# Champ de texte pour l'avis de l'utilisateur
user_review = st.text_area("Entrez votre avis", "")

# Bouton de prédiction
if st.button('Prédire'):
    sentiment = predict_sentiment(user_review)
    st.write(f"Le sentiment prédit est : {sentiment}")