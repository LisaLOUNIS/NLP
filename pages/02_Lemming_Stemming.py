import streamlit as st
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

# Titre de l'application
st.title("Stemming et Lemmatization en NLP")

# Description
st.write("""
    Cette application permet de comprendre la différence entre Stemming et Lemmatization.
    Entrez un texte et sélectionnez la méthode que vous souhaitez utiliser.
""")

# Champ de texte pour l'input utilisateur
text_input = st.text_input("Entrez un mot à traiter", "Stemming")

# Choix de la méthode
method = st.selectbox(
    "Choisissez une méthode",
    ["Stemming avec PorterStemmer", "Stemming avec LancasterStemmer", "Stemming avec SnowballStemmer", "Lemmatization"]
)

# Fonction pour effectuer le Stemming ou la Lemmatization
def process_text(text, method):
    if method == "Stemming avec PorterStemmer":
        stemmer = PorterStemmer()
        return stemmer.stem(text)
    elif method == "Stemming avec LancasterStemmer":
        stemmer = LancasterStemmer()
        return stemmer.stem(text)
    elif method == "Stemming avec SnowballStemmer":
        stemmer = SnowballStemmer("english")
        return stemmer.stem(text)
    elif method == "Lemmatization":
        lemmatizer = WordNetLemmatizer()
        return lemmatizer.lemmatize(text)

# Affichage du résultat
if text_input:
    st.write("Résultat :")
    st.write(process_text(text_input, method))
