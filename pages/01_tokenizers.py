# Import des bibliothèques nécessaires
import streamlit as st
from nltk.tokenize import word_tokenize, WhitespaceTokenizer, TreebankWordTokenizer

# Titre de l'application
st.title("Testeur de Tokenizers en NLP")

# Champ de texte pour l'input utilisateur
text_input = st.text_area("Entrez le texte à tokenizer", "")

# Choix du tokenizer
tokenizer_choice = st.selectbox(
    "Choisissez un tokenizer",
    ["Word Tokenize", "Whitespace Tokenize", "Treebank Word Tokenize"]
)

# Fonction pour effectuer le tokenizing
def tokenize_text(text, tokenizer):
    if tokenizer == "Word Tokenize":
        return word_tokenize(text)
    elif tokenizer == "Whitespace Tokenize":
        tokenizer = WhitespaceTokenizer()
        return tokenizer.tokenize(text)
    elif tokenizer == "Treebank Word Tokenize":
        tokenizer = TreebankWordTokenizer()
        return tokenizer.tokenize(text)

# Affichage du résultat
if text_input:
    st.write("Texte tokenisé :")
    st.write(tokenize_text(text_input, tokenizer_choice))
