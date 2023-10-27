# 1. Importez les modules nécessaires
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import treebank
from nltk.tag import UnigramTagger

# Téléchargez le Treebank et le modèle de POS tagging de NLTK
nltk.download('treebank')
nltk.download('punkt')

# Créez une instance du UnigramTagger en utilisant le Treebank comme données d'entraînement
train_sents = treebank.tagged_sents()[:3000]
tagger = UnigramTagger(train_sents)

# Créez une fonction pour effectuer le POS tagging
def pos_tag_treebank(text):
    tokenized_text = word_tokenize(text)
    return tagger.tag(tokenized_text)

# Utilisez Streamlit pour créer l'interface utilisateur
st.title("POS Tagging avec Streamlit, NLTK et Treebank")

# Zone de texte pour l'entrée utilisateur
user_input = st.text_area("Collez votre texte ici")

# Bouton pour déclencher le POS tagging
if st.button("Effectuer le POS Tagging"):
    tagged_text = pos_tag_treebank(user_input)
    st.write("### Résultat du POS Tagging")
    st.write(tagged_text)
